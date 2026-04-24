// P11 — K_smem Overwrite Race Probe
//
// Tests Hypothesis #6: tcgen05.commit's mbarrier arrival fires before MMA's
// source-SMEM reads are fully complete. If so, overwriting K_smem AFTER mbar.wait
// (which we believe means "MMA done") will cause MMA's still-in-flight tensor-core
// reads to pick up the new bytes, polluting MMA's output.
//
// Test sequence per CTA:
//   1. Load K_A (diag value 1.0) into K_smem
//   2. Dispatch MMA[A] → TMEM region 0 → commit mbar
//   3. mbar.wait  ← supposed to mean MMA done
//   4. (optional, mode-dependent) overwrite K_smem with K_C bytes (diag value 4.0)
//   5. tcgen05.ld region 0
//   6. Write final_scores per CTA — checked against expected
//
// Modes:
//   0 = NO overwrite (control: should match MMA[K_A] = 1.0 on diagonal)
//   1 = WITH overwrite (mbar correctly tracks MMA completion → still 1.0; if mbar
//       fires early → 4.0 on diagonal at the lane positions where MMA's reads were
//       not yet complete)
//
// Multi-CTA via grid (1, B). Each CTA does the same sequence; per-CTA final_scores
// slice should be the same (broadcast inputs). Multi-CTA stress reveals the bug
// because per-SM tensor pipe contention widens the gap between mbar arrival and
// actual MMA completion.

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

constexpr uint32_t IDESC_VAL  = (1u << 4) | (8u << 17) | (4u << 24);  // 0x04100010

constexpr int Q_SMEM_OFFSET      = 0;        // 8192 B
constexpr int K_SMEM_OFFSET      = 8192;     // 8192 B  (single buffer; overwritten in mode 1)
constexpr int ALLOC_SLOT_OFFSET  = 16384;    // 8 B
constexpr int MBAR_SLOT_OFFSET   = 16392;    // 8 B
constexpr int SMEM_BYTES         = 16400;

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

__device__ __forceinline__ void tcgen05_alloc_64(uint32_t alloc_slot_smem) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;"
                 :: "r"(alloc_slot_smem) : "memory");
}

__device__ __forceinline__ void tcgen05_dealloc_64(uint32_t taddr) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;"
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

__device__ __forceinline__ void tcgen05_ld_32x32b_x32(uint32_t taddr, uint32_t (&r)[32]) {
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

// =====================================================================
//  P11 kernel: K_smem overwrite race test.
//  One MMA per CTA. Optionally overwrite K_smem after mbar.wait.
//  Output is the raw MMA result for region 0 written to GMEM as FP32 [64, 64].
//  Each CTA writes its own slice (broadcast Q/K, but per-CTA out).
// =====================================================================
__global__ __launch_bounds__(128, 2)
void tmem2_p11_kernel(
    const uint8_t* __restrict__ q_fp8,         // [8192] FP8 8xT
    const uint8_t* __restrict__ k_a_fp8,       // [8192] FP8 8xT (value 1.0 diag)
    const uint8_t* __restrict__ k_c_fp8,       // [8192] FP8 8xT (value 4.0 diag) — overwrite source
    int            mode,                       // 0 = no overwrite, 1 = overwrite
    float*         __restrict__ out)           // [B * 64 * 64] FP32 — raw MMA output per CTA
{
    extern __shared__ __align__(16) uint8_t smem_raw[];

    uint8_t* Q_smem        = smem_raw + Q_SMEM_OFFSET;
    uint8_t* K_smem        = smem_raw + K_SMEM_OFFSET;
    uint32_t* alloc_slot_ptr = reinterpret_cast<uint32_t*>(smem_raw + ALLOC_SLOT_OFFSET);

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int b       = blockIdx.y;

    const uint32_t q_shared    = __cvta_generic_to_shared(Q_smem);
    const uint32_t k_shared    = __cvta_generic_to_shared(K_smem);
    const uint32_t alloc_slot_s = __cvta_generic_to_shared(alloc_slot_ptr);
    const uint32_t mbar_s      = __cvta_generic_to_shared(smem_raw + MBAR_SLOT_OFFSET);

    if (warp_id == 1) tcgen05_alloc_64(alloc_slot_s);
    __syncthreads();
    const uint32_t tmem_col = alloc_slot_ptr[0];

    if (warp_id == 0 && lane_id == 0) mbarrier_init_1(mbar_s);
    __syncthreads();

    // Load Q via direct uint4
    {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int u   = tid * 4 + i;
            int off = u << 4;
            *reinterpret_cast<uint4*>(Q_smem + off) =
                *reinterpret_cast<const uint4*>(q_fp8 + off);
        }
    }

    // Load K_A into K_smem (the K we want MMA to use)
    {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int u   = tid * 4 + i;
            int off = u << 4;
            *reinterpret_cast<uint4*>(K_smem + off) =
                *reinterpret_cast<const uint4*>(k_a_fp8 + off);
        }
    }
    __syncthreads();

    // Dispatch MMA (using K_smem = K_A) → region 0 → commit mbar
    if (warp_id == 0 && elect_one_sync()) {
        #pragma unroll 1
        for (int s = 0; s < NUM_SLABS; ++s) {
            const uint32_t q_slab_addr = q_shared + s * SLAB_BYTES;
            const uint32_t k_slab_addr = k_shared + s * SLAB_BYTES;
            const uint64_t a_desc = make_smem_desc(q_slab_addr);
            const uint64_t b_desc = make_smem_desc(k_slab_addr);
            const int enable_d = (s != 0) ? 1 : 0;
            tcgen05_mma_f8f6f4(tmem_col, a_desc, b_desc, IDESC_VAL, enable_d);
        }
        tcgen05_commit_mbar(mbar_s);
    }

    // Wait MMA — supposed to mean MMA's TMEM writes done AND its SMEM reads done
    mbarrier_wait_phase(mbar_s, 0);
    tcgen05_fence_after_sync();

    // === THE TEST: overwrite K_smem with K_C bytes (value 4.0) ===
    // If mbar fires before MMA's K_smem reads complete, MMA's still-in-flight reads
    // will pick up K_C bytes and produce K_C-flavored output (4.0 on diagonal).
    if (mode == 1) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int u   = tid * 4 + i;
            int off = u << 4;
            *reinterpret_cast<uint4*>(K_smem + off) =
                *reinterpret_cast<const uint4*>(k_c_fp8 + off);
        }
        __syncthreads();
    }

    // Read TMEM region 0 → GMEM
    const uint32_t taddr_lo = (uint32_t(warp_id) * 32u) << 16 | tmem_col;
    const uint32_t taddr_hi = (uint32_t(warp_id) * 32u) << 16 | (tmem_col + 32u);
    uint32_t rlo[32], rhi[32];
    tcgen05_ld_32x32b_x32(taddr_lo, rlo);
    tcgen05_ld_32x32b_x32(taddr_hi, rhi);
    tcgen05_wait_ld();

    if (lane_id < 16) {
        const int row = warp_id * 16 + lane_id;     // 0..63
        float* row_out = out + b * 64 * 64 + row * 64;
        #pragma unroll
        for (int c = 0; c < 32; ++c) {
            row_out[c]      = __uint_as_float(rlo[c]);
            row_out[c + 32] = __uint_as_float(rhi[c]);
        }
    }

    __syncthreads();
    if (warp_id == 1) tcgen05_dealloc_64(tmem_col);
}

// ===== Host launcher =====
void run_p11(
    torch::Tensor q_fp8,        // uint8 [8192]
    torch::Tensor k_a_fp8,      // uint8 [8192]
    torch::Tensor k_c_fp8,      // uint8 [8192]
    int64_t       mode,         // 0 = control, 1 = with K_smem overwrite
    torch::Tensor out,          // float32 [B * 64 * 64]
    int64_t B)
{
    const at::cuda::CUDAGuard device_guard(q_fp8.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    dim3 grid(1, static_cast<int>(B));
    tmem2_p11_kernel<<<grid, 128, SMEM_BYTES, stream>>>(
        q_fp8.data_ptr<uint8_t>(),
        k_a_fp8.data_ptr<uint8_t>(),
        k_c_fp8.data_ptr<uint8_t>(),
        static_cast<int>(mode),
        out.data_ptr<float>());
    AT_CUDA_CHECK(cudaGetLastError());
}
