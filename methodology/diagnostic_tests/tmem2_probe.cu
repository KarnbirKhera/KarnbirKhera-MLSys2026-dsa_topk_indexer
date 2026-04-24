// TMEM Double-Buffer Diagnostic Probe
// Isolates GT-47's failure mode: which exact mechanism breaks when running
// MMA[A] and MMA[B] in flight to TMEM regions A=tmem_col, B=tmem_col+64.
//
// Test data (hand-crafted so expected output is bit-exact computable):
//   Q[m, k]   = 1.0 if (m == k && m < 64) else 0.0     // diagonal in [0,64)x[0,128)
//   K_A[n, k] = 1.0 if (n == k && n < 64) else 0.0     // diagonal
//   K_B[n, k] = 2.0 if (n == k && n < 64) else 0.0     // 2x diagonal
//   K_C[n, k] = 4.0 if (n == k && n < 64) else 0.0     // 4x diagonal (P4/P5)
//
// Expected MMA results (D = Q · K^T):
//   D_A[m, n] = 1.0 if m == n else 0.0      (64x64 identity)
//   D_B[m, n] = 2.0 if m == n else 0.0      (2x identity)
//   D_C[m, n] = 4.0 if m == n else 0.0      (4x identity)
//
// Probe modes (passed as `mode` arg):
//   0 = M0 baseline:  single MMA to region A only.  Sanity check.
//   1 = P1:           single MMA to region B only.  Tests tmem_col+64 addressing in isolation.
//   2 = P2:           dispatch MMA[A]+commit, then MMA[B]+commit (no wait between);
//                     wait A → ld A → store; wait B → ld B → store.  Tests two-in-flight pattern.
//   3 = P3:           same as P2 but FORCE a single specific lane (lane 0 of warp 0)
//                     to dispatch BOTH MMAs and BOTH commits.  Eliminates elect_one_sync
//                     non-determinism as a variable.
//   4 = P4:           same mbar used twice in sequence: MMA[A_K=1.0]→commit mbar_A→
//                     wait mbar_A (phase=0) → ld region A (verify D_A=I) → MMA[A_K=4.0]→
//                     commit mbar_A → wait mbar_A (phase=1) → ld region A (verify D_C=4I).
//                     Tests mbar reuse with phase-parity flip.
//   5 = P5:           production-like 3-MMA flow:
//                       MMA[A,K=1] → mbar_A
//                       MMA[B,K=2] → mbar_B
//                       wait mbar_A (phase=0) → ld region A → out_a
//                       MMA[A,K=4] → mbar_A      (region A REUSED + mbar_A REUSED)
//                       wait mbar_B (phase=0) → ld region B → out_b
//                       wait mbar_A (phase=1) → ld region A → out_c
//                     Tests mbar reuse + region overwrite + the EXACT pattern in
//                     the production loop where iter pidx+1 dispatches MMA to a
//                     region that had a prior MMA, with mbar phase tracking.

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

constexpr int INDEX_HEAD_DIM  = 128;
constexpr int PAGE_SIZE       = 64;
constexpr int H               = 64;
constexpr int MMA_M           = 64;
constexpr int MMA_N           = 64;
constexpr int MMA_K           = 32;
constexpr int NUM_SLABS       = INDEX_HEAD_DIM / MMA_K;  // 4

// IDESC: M=64, N=64, K=32, dtype=F32, atype=E4M3, btype=E4M3, no transpose.
constexpr uint32_t IDESC_VAL  = (1u << 4) | (8u << 17) | (4u << 24);  // 0x04100010

// SMEM layout (8-B aligned for mbars; 16-B aligned for tiles)
constexpr int Q_SMEM_OFFSET      = 0;
constexpr int K_A_SMEM_OFFSET    = 8192;
constexpr int K_B_SMEM_OFFSET    = 16384;
constexpr int K_C_SMEM_OFFSET    = 24576;   // 8192 B (P4/P5)
constexpr int ALLOC_SLOT_OFFSET  = 32768;   // 8 B
constexpr int MBAR_A_SLOT_OFFSET = 32776;   // 8 B
constexpr int MBAR_B_SLOT_OFFSET = 32784;   // 8 B
constexpr int SMEM_BYTES         = 32792;

// 8xT core-tile constants
constexpr int SLAB_BYTES = 2048;
constexpr int SBO_BYTES  = 256;
constexpr int LBO_BYTES  = 128;

__device__ __forceinline__ int smem_8xT_offset(int m, int k) {
    return (k / 32) * SLAB_BYTES
         + (m / 8)  * SBO_BYTES
         + ((k % 32) / 16) * LBO_BYTES
         + (m % 8)  * 16
         + (k % 16);
}

__device__ __forceinline__ uint64_t desc_encode_u64(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}

__device__ __forceinline__ uint64_t make_smem_desc(uint32_t smem_addr_shared) {
    uint64_t d = 0;
    d |= desc_encode_u64((uint64_t)smem_addr_shared);
    d |= desc_encode_u64((uint64_t)LBO_BYTES) << 16;
    d |= desc_encode_u64((uint64_t)SBO_BYTES) << 32;
    d |= (uint64_t)0b001ULL << 46;
    return d;
}

__device__ __forceinline__ uint32_t elect_one_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        " .reg .pred %%px;\n\t"
        " elect.sync _|%%px, %1;\n\t"
        " @%%px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred) : "r"(0xFFFFFFFFu)
    );
    return pred;
}

__device__ __forceinline__ void mbarrier_init_1(uint32_t mbar_smem) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                 :: "r"(mbar_smem) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_phase(uint32_t mbar_smem, uint32_t phase) {
    uint32_t ticks = 0x989680u;
    asm volatile(
        "{\n\t"
        " .reg .pred P1;\n\t"
        "LAB_WAIT_%=:\n\t"
        " mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        " @P1 bra DONE_%=;\n\t"
        " bra LAB_WAIT_%=;\n\t"
        "DONE_%=:\n\t"
        "}"
        :: "r"(mbar_smem), "r"(phase), "r"(ticks)
    );
}

__device__ __forceinline__ void tcgen05_alloc_128(uint32_t alloc_slot_smem) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;"
                 :: "r"(alloc_slot_smem) : "memory");
}

__device__ __forceinline__ void tcgen05_dealloc_128(uint32_t taddr) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;"
                 :: "r"(taddr) : "memory");
}

__device__ __forceinline__ void tcgen05_commit_mbar(uint32_t mbar_smem) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(mbar_smem) : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    uint32_t tmem_addr, uint64_t a_desc, uint64_t b_desc,
    uint32_t idesc, int enable_input_d)
{
    asm volatile(
        "{\n\t"
        " .reg .pred p;\n\t"
        " setp.ne.b32 p, %4, 0;\n\t"
        " tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
        "}"
        :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc),
           "r"(idesc), "r"(enable_input_d)
    );
}

__device__ __forceinline__ void tcgen05_ld_32x32b_x32(
    uint32_t taddr,
    uint32_t (&r)[32])
{
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        " %8,%9,%10,%11,%12,%13,%14,%15,"
        " %16,%17,%18,%19,%20,%21,%22,%23,"
        " %24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(r[0]),  "=r"(r[1]),  "=r"(r[2]),  "=r"(r[3]),
          "=r"(r[4]),  "=r"(r[5]),  "=r"(r[6]),  "=r"(r[7]),
          "=r"(r[8]),  "=r"(r[9]),  "=r"(r[10]), "=r"(r[11]),
          "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
          "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),
          "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
          "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
          "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
        : "r"(taddr)
    );
}

// Issues a 4-slab MMA from given Q_smem and K_smem bases into the given TMEM column.
// Caller is responsible for surrounding it with the right elect-thread guards.
__device__ __forceinline__ void issue_mma_4slab(
    uint32_t q_shared, uint32_t k_shared, uint32_t tmem_addr)
{
    #pragma unroll 1
    for (int s = 0; s < NUM_SLABS; ++s) {
        const uint32_t q_slab_addr = q_shared + s * SLAB_BYTES;
        const uint32_t k_slab_addr = k_shared + s * SLAB_BYTES;
        const uint64_t a_desc = make_smem_desc(q_slab_addr);
        const uint64_t b_desc = make_smem_desc(k_slab_addr);
        const int enable_d = (s != 0) ? 1 : 0;
        tcgen05_mma_f8f6f4(tmem_addr, a_desc, b_desc, IDESC_VAL, enable_d);
    }
}

// Read TMEM region at column base and store into out [64x64 FP32].
// All warps participate; only lane<16 of each warp writes (matches GT-12 distribution).
__device__ __forceinline__ void read_tmem_to_gmem(
    uint32_t tmem_col_base, float* __restrict__ out, int warp_id, int lane_id)
{
    const uint32_t taddr_lo = (uint32_t(warp_id) * 32u) << 16 | tmem_col_base;
    const uint32_t taddr_hi = (uint32_t(warp_id) * 32u) << 16 | (tmem_col_base + 32u);
    uint32_t rlo[32], rhi[32];
    tcgen05_ld_32x32b_x32(taddr_lo, rlo);
    tcgen05_ld_32x32b_x32(taddr_hi, rhi);
    tcgen05_wait_ld();
    if (lane_id < 16) {
        const int row = warp_id * 16 + lane_id;     // 0..63
        float* row_out = out + row * 64;
        #pragma unroll
        for (int c = 0; c < 32; ++c) {
            row_out[c]      = __uint_as_float(rlo[c]);
            row_out[c + 32] = __uint_as_float(rhi[c]);
        }
    }
}

// Single-CTA, 128-thread diagnostic kernel.
// Inputs are 8KB FP8 tiles already in 8xT byte layout (host fills them).
// Outputs are FP32 [64, 64]; out_a/out_b/out_c receive results (or stay 0 if unused).
__global__ __launch_bounds__(128, 1)
void tmem2_probe_kernel(
    const uint8_t* __restrict__ q_fp8,        // [64, 128] FP8 in 8xT layout
    const uint8_t* __restrict__ k_a_fp8,      // [64, 128] FP8 in 8xT layout (1.0 diag)
    const uint8_t* __restrict__ k_b_fp8,      // [64, 128] FP8 in 8xT layout (2.0 diag)
    const uint8_t* __restrict__ k_c_fp8,      // [64, 128] FP8 in 8xT layout (4.0 diag) — P4/P5
    float*         __restrict__ out_a,        // [64, 64] FP32
    float*         __restrict__ out_b,        // [64, 64] FP32
    float*         __restrict__ out_c,        // [64, 64] FP32 (P5 only)
    int mode)
{
    extern __shared__ __align__(16) uint8_t smem_raw[];

    uint8_t* Q_smem   = smem_raw + Q_SMEM_OFFSET;
    uint8_t* K_A_smem = smem_raw + K_A_SMEM_OFFSET;
    uint8_t* K_B_smem = smem_raw + K_B_SMEM_OFFSET;
    uint8_t* K_C_smem = smem_raw + K_C_SMEM_OFFSET;
    uint32_t* alloc_slot_ptr = reinterpret_cast<uint32_t*>(smem_raw + ALLOC_SLOT_OFFSET);

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const uint32_t q_shared    = __cvta_generic_to_shared(Q_smem);
    const uint32_t k_a_shared  = __cvta_generic_to_shared(K_A_smem);
    const uint32_t k_b_shared  = __cvta_generic_to_shared(K_B_smem);
    const uint32_t k_c_shared  = __cvta_generic_to_shared(K_C_smem);
    const uint32_t alloc_slot_s = __cvta_generic_to_shared(alloc_slot_ptr);
    const uint32_t mbar_a_s    = __cvta_generic_to_shared(smem_raw + MBAR_A_SLOT_OFFSET);
    const uint32_t mbar_b_s    = __cvta_generic_to_shared(smem_raw + MBAR_B_SLOT_OFFSET);

    // --- TMEM alloc 128 cols ---
    if (warp_id == 1) {
        tcgen05_alloc_128(alloc_slot_s);
    }
    __syncthreads();
    const uint32_t tmem_col_base = alloc_slot_ptr[0];
    const uint32_t tmem_col_A = tmem_col_base;
    const uint32_t tmem_col_B = tmem_col_base + 64u;

    // --- mbar init ---
    if (warp_id == 0 && lane_id == 0) mbarrier_init_1(mbar_a_s);
    if (warp_id == 0 && lane_id == 1) mbarrier_init_1(mbar_b_s);
    __syncthreads();

    // --- Cooperative load Q, K_A, K_B, K_C from GMEM to SMEM (8xT byte-for-byte) ---
    // 8192 B per tile = 512 uint4 per tile; 128 threads × 4 uint4 each.
    {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int u   = tid * 4 + i;
            int off = u << 4;   // 16 B per chunk
            *reinterpret_cast<uint4*>(Q_smem   + off) = *reinterpret_cast<const uint4*>(q_fp8   + off);
            *reinterpret_cast<uint4*>(K_A_smem + off) = *reinterpret_cast<const uint4*>(k_a_fp8 + off);
            *reinterpret_cast<uint4*>(K_B_smem + off) = *reinterpret_cast<const uint4*>(k_b_fp8 + off);
            *reinterpret_cast<uint4*>(K_C_smem + off) = *reinterpret_cast<const uint4*>(k_c_fp8 + off);
        }
    }
    __syncthreads();

    // --- Zero-init outputs (so unused regions are obviously 0) ---
    if (tid < 64) {
        for (int c = 0; c < 64; ++c) {
            out_a[tid * 64 + c] = 0.0f;
            out_b[tid * 64 + c] = 0.0f;
            out_c[tid * 64 + c] = 0.0f;
        }
    }
    __syncthreads();

    uint32_t phase_a = 0, phase_b = 0;

    if (mode == 0) {
        // ===== M0: single MMA to region A. Baseline sanity check. =====
        if (warp_id == 0 && elect_one_sync()) {
            issue_mma_4slab(q_shared, k_a_shared, tmem_col_A);
            tcgen05_commit_mbar(mbar_a_s);
        }
        mbarrier_wait_phase(mbar_a_s, phase_a);
        tcgen05_fence_after_sync();
        read_tmem_to_gmem(tmem_col_A, out_a, warp_id, lane_id);
    }
    else if (mode == 1) {
        // ===== P1: single MMA to region B (tmem_col + 64). =====
        if (warp_id == 0 && elect_one_sync()) {
            issue_mma_4slab(q_shared, k_b_shared, tmem_col_B);
            tcgen05_commit_mbar(mbar_b_s);
        }
        mbarrier_wait_phase(mbar_b_s, phase_b);
        tcgen05_fence_after_sync();
        read_tmem_to_gmem(tmem_col_B, out_b, warp_id, lane_id);
    }
    else if (mode == 2) {
        // ===== P2: dispatch MMA[A] then MMA[B] (no wait between), then ld both. =====
        if (warp_id == 0 && elect_one_sync()) {
            issue_mma_4slab(q_shared, k_a_shared, tmem_col_A);
            tcgen05_commit_mbar(mbar_a_s);
        }
        if (warp_id == 0 && elect_one_sync()) {
            issue_mma_4slab(q_shared, k_b_shared, tmem_col_B);
            tcgen05_commit_mbar(mbar_b_s);
        }
        // Wait A → ld A → wait B → ld B
        mbarrier_wait_phase(mbar_a_s, phase_a);
        tcgen05_fence_after_sync();
        read_tmem_to_gmem(tmem_col_A, out_a, warp_id, lane_id);

        mbarrier_wait_phase(mbar_b_s, phase_b);
        tcgen05_fence_after_sync();
        read_tmem_to_gmem(tmem_col_B, out_b, warp_id, lane_id);
    }
    else if (mode == 3) {
        // ===== P3: P2 with FORCED single thread (lane 0 of warp 0) for both MMAs. =====
        if (warp_id == 0 && lane_id == 0) {
            issue_mma_4slab(q_shared, k_a_shared, tmem_col_A);
            tcgen05_commit_mbar(mbar_a_s);
            issue_mma_4slab(q_shared, k_b_shared, tmem_col_B);
            tcgen05_commit_mbar(mbar_b_s);
        }
        mbarrier_wait_phase(mbar_a_s, phase_a);
        tcgen05_fence_after_sync();
        read_tmem_to_gmem(tmem_col_A, out_a, warp_id, lane_id);

        mbarrier_wait_phase(mbar_b_s, phase_b);
        tcgen05_fence_after_sync();
        read_tmem_to_gmem(tmem_col_B, out_b, warp_id, lane_id);
    }
    else if (mode == 4) {
        // ===== P4: same mbar reused twice — commit/wait/commit/wait pattern. =====
        // out_a receives the MMA[K_A] result (D = I)
        // out_c receives the MMA[K_C] result (D = 4*I) using region A REUSED
        if (warp_id == 0 && elect_one_sync()) {
            issue_mma_4slab(q_shared, k_a_shared, tmem_col_A);
            tcgen05_commit_mbar(mbar_a_s);
        }
        mbarrier_wait_phase(mbar_a_s, phase_a);
        phase_a ^= 1;
        tcgen05_fence_after_sync();
        read_tmem_to_gmem(tmem_col_A, out_a, warp_id, lane_id);

        // Need a barrier so all warps finished reading region A before next MMA writes it.
        __syncthreads();

        if (warp_id == 0 && elect_one_sync()) {
            issue_mma_4slab(q_shared, k_c_shared, tmem_col_A);
            tcgen05_commit_mbar(mbar_a_s);
        }
        mbarrier_wait_phase(mbar_a_s, phase_a);
        tcgen05_fence_after_sync();
        read_tmem_to_gmem(tmem_col_A, out_c, warp_id, lane_id);
    }
    else if (mode == 5) {
        // ===== P5: production-like 3-MMA flow with mbar_A reused. =====
        // Mirrors the production loop's structure where iter pidx+1's stage A
        // dispatches into the same TMEM region + same mbar that iter pidx-1 used.
        //
        // Sequence:
        //   MMA[A, K_A=1]  → mbar_A
        //   MMA[B, K_B=2]  → mbar_B
        //   wait mbar_A → ld region A → out_a       (expect D = I)
        //   MMA[A, K_C=4]  → mbar_A   (REGION A REUSED, mbar_A REUSED)
        //   wait mbar_B → ld region B → out_b       (expect D = 2I)
        //   wait mbar_A → ld region A → out_c       (expect D = 4I)
        if (warp_id == 0 && elect_one_sync()) {
            issue_mma_4slab(q_shared, k_a_shared, tmem_col_A);
            tcgen05_commit_mbar(mbar_a_s);
        }
        if (warp_id == 0 && elect_one_sync()) {
            issue_mma_4slab(q_shared, k_b_shared, tmem_col_B);
            tcgen05_commit_mbar(mbar_b_s);
        }

        // Wait A → ld → store
        mbarrier_wait_phase(mbar_a_s, phase_a);
        phase_a ^= 1;
        tcgen05_fence_after_sync();
        read_tmem_to_gmem(tmem_col_A, out_a, warp_id, lane_id);

        // Barrier so all warps done with region A before MMA[A2] writes it.
        __syncthreads();

        // Re-dispatch MMA into region A with K_C, signal mbar_A (now phase=1).
        // MMA[B] is still in flight.
        if (warp_id == 0 && elect_one_sync()) {
            issue_mma_4slab(q_shared, k_c_shared, tmem_col_A);
            tcgen05_commit_mbar(mbar_a_s);
        }

        // Wait B → ld → store
        mbarrier_wait_phase(mbar_b_s, phase_b);
        tcgen05_fence_after_sync();
        read_tmem_to_gmem(tmem_col_B, out_b, warp_id, lane_id);

        // Wait A (second use, phase=1) → ld → store
        mbarrier_wait_phase(mbar_a_s, phase_a);
        tcgen05_fence_after_sync();
        read_tmem_to_gmem(tmem_col_A, out_c, warp_id, lane_id);
    }

    __syncthreads();

    // --- TMEM dealloc ---
    if (warp_id == 1) {
        tcgen05_dealloc_128(tmem_col_base);
    }
}

// ===== Host launcher (torch ext entry point) =====
void run_probe(
    torch::Tensor q_fp8,        // uint8 [8192]
    torch::Tensor k_a_fp8,      // uint8 [8192]
    torch::Tensor k_b_fp8,      // uint8 [8192]
    torch::Tensor k_c_fp8,      // uint8 [8192]   — P4/P5 only
    torch::Tensor out_a,        // float32 [64, 64]
    torch::Tensor out_b,        // float32 [64, 64]
    torch::Tensor out_c,        // float32 [64, 64] — P4/P5 only
    int64_t mode)
{
    const at::cuda::CUDAGuard device_guard(q_fp8.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const uint8_t* q_ptr   = q_fp8.data_ptr<uint8_t>();
    const uint8_t* k_a_ptr = k_a_fp8.data_ptr<uint8_t>();
    const uint8_t* k_b_ptr = k_b_fp8.data_ptr<uint8_t>();
    const uint8_t* k_c_ptr = k_c_fp8.data_ptr<uint8_t>();
    float* a_ptr = out_a.data_ptr<float>();
    float* b_ptr = out_b.data_ptr<float>();
    float* c_ptr = out_c.data_ptr<float>();

    tmem2_probe_kernel<<<1, 128, SMEM_BYTES, stream>>>(
        q_ptr, k_a_ptr, k_b_ptr, k_c_ptr,
        a_ptr, b_ptr, c_ptr,
        static_cast<int>(mode));
    AT_CUDA_CHECK(cudaGetLastError());
}

// PYBIND11_MODULE removed — torch.utils.cpp_extension.load_inline auto-generates
// the binding from the `functions` arg in run_tmem2_probe.py.
