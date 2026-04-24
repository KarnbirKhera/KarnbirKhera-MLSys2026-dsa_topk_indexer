// P6 — TMEM2 Production-Replica Probe
//
// Mirrors the failing production buggy loop body (Stage A reorder + H-reduce +
// scale apply + scores_smem + GMEM write) using single CTA + synthetic data so
// expected outputs are bit-exact verifiable. If P6 reproduces the GT-47 bug, we
// have a minimal, controllable repro to instrument further (Option A). If P6
// passes, the bug requires multi-batch/multi-CTA parallelism we can't isolate.
//
// Test data (each tile in 8xT FP8 layout, hand-built host-side):
//   Q[m, k]   = 1.0 if (m == k && m < 64) else 0.0     // diagonal
//   K_p[n, k] = value_p if (n == k && n < 64) else 0.0 // diagonal, value_p ∈ {1,2,4,8}
//   weights[h] = 1.0  (FP32)
//   token_scale_p = 1.0  (FP32 in scale_smem buffer)
//
// MMA (FP8 e4m3 inputs, FP32 accumulator):
//   D_p[m, n] = sum_k Q[m,k] * K_p[n,k] = value_p if m == n else 0
//
// Scale apply (token-scale = 1):
//   scaled_scores_p[h, n] = D_p[h, n] = value_p if h == n else 0
//
// H-reduce (relu × weight, weight = 1):
//   final_scores_p[n] = sum_h relu(scaled_scores_p[h, n]) * 1
//                      = sum_h (value_p if h == n else 0)
//                      = value_p   for all n in [0, 64)
//
// Expected final_scores layout [page * 64 + col]:
//   [0..63]   = 1.0
//   [64..127] = 2.0
//   [128..191] = 4.0
//   [192..255] = 8.0

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
constexpr int PAGES_PER_CTA   = 4;
constexpr int MMA_M           = 64;
constexpr int MMA_N           = 64;
constexpr int MMA_K           = 32;
constexpr int NUM_SLABS       = INDEX_HEAD_DIM / MMA_K;  // 4
constexpr int SCORES_STRIDE   = 65;
constexpr uint32_t IDESC_VAL  = (1u << 4) | (8u << 17) | (4u << 24);  // 0x04100010

// SMEM layout matches production buggy kernel; P10 adds 2 more mbar slots.
constexpr int Q_SMEM_OFFSET       = 0;                    // 8192 B
constexpr int K0_SMEM_OFFSET      = 8192;                 // 8192 B
constexpr int K1_SMEM_OFFSET      = 16384;                // 8192 B
constexpr int SCORES_SMEM_OFFSET  = 24576;                // 16640 B (64 × 65 FP32)
constexpr int SCALE0_SMEM_OFFSET  = 41216;                // 256 B
constexpr int SCALE1_SMEM_OFFSET  = 41472;                // 256 B
constexpr int ALLOC_SLOT_OFFSET   = 41728;                // 8 B
constexpr int MBAR_A_SLOT_OFFSET  = 41736;                // 8 B (page 0)
constexpr int MBAR_B_SLOT_OFFSET  = 41744;                // 8 B (page 1)
constexpr int MBAR_C_SLOT_OFFSET  = 41752;                // 8 B (page 2 — P10)
constexpr int MBAR_D_SLOT_OFFSET  = 41760;                // 8 B (page 3 — P10)
constexpr int WEIGHTS_SMEM_OFFSET = 41776;                // 256 B (16-B aligned)
constexpr int SMEM_BYTES          = 42032;

// 8xT core-tile constants (GT-11)
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

// P10: alloc 256 cols (4 regions × 64 cols each, no reuse across pages).
__device__ __forceinline__ void tcgen05_alloc_256(uint32_t alloc_slot_smem) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;"
                 :: "r"(alloc_slot_smem) : "memory");
}

__device__ __forceinline__ void tcgen05_dealloc_256(uint32_t taddr) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;"
                 :: "r"(taddr) : "memory");
}

__device__ __forceinline__ void tcgen05_commit_mbar(uint32_t mbar_smem) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(mbar_smem) : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// P8 FIX: needed after wait::ld to order ld with subsequent cross-thread MMA dispatch
// per PTX §9.7.16.6.4.4 (non-pipelined ld → mma cross-thread to same accumulator).
__device__ __forceinline__ void tcgen05_fence_before_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
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
    uint32_t taddr, uint32_t (&r)[32])
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

// =====================================================================
//  P6 production-replica kernel.
//  Single CTA, 128 threads. Replicates the production buggy loop body
//  EXACTLY (Stage A reorder, scale apply, H-reduce, GMEM final_scores).
//  Inputs: Q (8KB), K_pages[4][8KB] in GMEM (8xT layout), weights[64], token_scale.
//  Output: final_scores[256] FP32  (4 pages × 64 cols).
// =====================================================================
// P7 extension: launch_bounds(128, 2) to match production occupancy. Each CTA's
// blockIdx.y selects which final_scores slice it writes to (one per "batch" CTA).
// Q and k_pages_fp8 are SHARED across all CTAs (read-only). final_scores is
// [B × 256] — each CTA writes its own 256-float slice.
__global__ __launch_bounds__(128, 2)
void tmem2_p6_kernel(
    const uint8_t* __restrict__ q_fp8,           // [8192] FP8 8xT
    const uint8_t* __restrict__ k_pages_fp8,     // [4 × 8192] FP8 8xT, page-major
    const float*   __restrict__ weights,         // [64] FP32
    float          token_scale,                  // applied to all tokens (uniform)
    float*         __restrict__ final_scores)    // [B × 256] FP32 — per-CTA slice
{
    extern __shared__ __align__(16) uint8_t smem_raw[];

    uint8_t* Q_smem        = smem_raw + Q_SMEM_OFFSET;
    uint8_t* K_smem_buf[2] = { smem_raw + K0_SMEM_OFFSET,
                                smem_raw + K1_SMEM_OFFSET };
    float*   scores_smem   = reinterpret_cast<float*>(smem_raw + SCORES_SMEM_OFFSET);
    float*   scale_smem_buf[2] = {
        reinterpret_cast<float*>(smem_raw + SCALE0_SMEM_OFFSET),
        reinterpret_cast<float*>(smem_raw + SCALE1_SMEM_OFFSET) };
    uint32_t* alloc_slot_ptr = reinterpret_cast<uint32_t*>(smem_raw + ALLOC_SLOT_OFFSET);
    float*   weights_smem    = reinterpret_cast<float*>(smem_raw + WEIGHTS_SMEM_OFFSET);

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const uint32_t q_shared       = __cvta_generic_to_shared(Q_smem);
    const uint32_t k_shared_buf[2] = {
        __cvta_generic_to_shared(K_smem_buf[0]),
        __cvta_generic_to_shared(K_smem_buf[1]) };
    const uint32_t alloc_slot_s   = __cvta_generic_to_shared(alloc_slot_ptr);
    // P10: 4 mbarriers (one per page, no reuse across pages)
    const uint32_t mbar_s_buf[4] = {
        __cvta_generic_to_shared(smem_raw + MBAR_A_SLOT_OFFSET),
        __cvta_generic_to_shared(smem_raw + MBAR_B_SLOT_OFFSET),
        __cvta_generic_to_shared(smem_raw + MBAR_C_SLOT_OFFSET),
        __cvta_generic_to_shared(smem_raw + MBAR_D_SLOT_OFFSET) };

    // ---- P10: TMEM alloc 256 cols (4 regions × 64 cols, no reuse across pages) ----
    if (warp_id == 1) tcgen05_alloc_256(alloc_slot_s);
    __syncthreads();
    const uint32_t tmem_col_base = alloc_slot_ptr[0];
    const uint32_t tmem_col_buf[4] = {
        tmem_col_base + 0u,
        tmem_col_base + 64u,
        tmem_col_base + 128u,
        tmem_col_base + 192u };

    // ---- mbar init (4 mbars) + weights load ----
    if (warp_id == 0 && lane_id == 0) mbarrier_init_1(mbar_s_buf[0]);
    if (warp_id == 0 && lane_id == 1) mbarrier_init_1(mbar_s_buf[1]);
    if (warp_id == 0 && lane_id == 2) mbarrier_init_1(mbar_s_buf[2]);
    if (warp_id == 0 && lane_id == 3) mbarrier_init_1(mbar_s_buf[3]);
    if (tid < H) weights_smem[tid] = weights[tid];
    uint32_t phase_buf[4] = { 0, 0, 0, 0 };
    __syncthreads();

    // ---- Q load via direct uint4 copy (matches probe convention; production uses cp.async,
    //      but we already showed cp.async is not the cause via bisect (e). Use sync for clarity.) ----
    {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int u   = tid * 4 + i;
            int off = u << 4;
            *reinterpret_cast<uint4*>(Q_smem + off) =
                *reinterpret_cast<const uint4*>(q_fp8 + off);
        }
    }

    // ---- Prologue: load page 0 + page 1 K tiles into K_smem buffers + scale ----
    {
        // Page 0 → K_smem[0]
        const uint8_t* k0_src = k_pages_fp8 + 0 * 8192;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int u   = tid * 4 + i;
            int off = u << 4;
            *reinterpret_cast<uint4*>(K_smem_buf[0] + off) =
                *reinterpret_cast<const uint4*>(k0_src + off);
        }
        if (tid < PAGE_SIZE) scale_smem_buf[0][tid] = token_scale;

        // Page 1 → K_smem[1]
        const uint8_t* k1_src = k_pages_fp8 + 1 * 8192;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int u   = tid * 4 + i;
            int off = u << 4;
            *reinterpret_cast<uint4*>(K_smem_buf[1] + off) =
                *reinterpret_cast<const uint4*>(k1_src + off);
        }
        if (tid < PAGE_SIZE) scale_smem_buf[1][tid] = token_scale;
    }
    __syncthreads();

    // ---- Prologue: dispatch MMA[0] into TMEM region 0, signal mbar[0] ----
    if (warp_id == 0 && elect_one_sync()) {
        const uint32_t tmem_addr0 = tmem_col_buf[0];
        #pragma unroll 1
        for (int s = 0; s < NUM_SLABS; ++s) {
            const uint32_t q_slab_addr = q_shared + s * SLAB_BYTES;
            const uint32_t k_slab_addr = k_shared_buf[0] + s * SLAB_BYTES;
            const uint64_t a_desc = make_smem_desc(q_slab_addr);
            const uint64_t b_desc = make_smem_desc(k_slab_addr);
            const int enable_d = (s != 0) ? 1 : 0;
            tcgen05_mma_f8f6f4(tmem_addr0, a_desc, b_desc, IDESC_VAL, enable_d);
        }
        tcgen05_commit_mbar(mbar_s_buf[0]);
    }

    // ---- P10: 4 TMEM regions, no reuse — each page p uses tmem_col + p*64 and mbar[p] ----
    constexpr int num_active = PAGES_PER_CTA;
    for (int pidx = 0; pidx < num_active; ++pidx) {
        const int curr_tmem  = pidx;          // P10: was pidx & 1
        const int next_pidx  = pidx + 1;
        const int later_pidx = pidx + 2;
        const bool has_next  = (next_pidx  < num_active);
        const bool has_later = (later_pidx < num_active);

        // STAGE A: dispatch MMA[next_pidx] into TMEM region next_pidx (own region)
        if (has_next) {
            if (warp_id == 0 && elect_one_sync()) {
                tcgen05_fence_after_sync();
                const int next_tmem  = next_pidx;     // P10: was next_pidx & 1
                const int next_K_buf = next_pidx & 1; // K_smem still double-buffered
                const uint32_t tmem_addr_n = tmem_col_buf[next_tmem];
                #pragma unroll 1
                for (int s = 0; s < NUM_SLABS; ++s) {
                    const uint32_t q_slab_addr = q_shared + s * SLAB_BYTES;
                    const uint32_t k_slab_addr = k_shared_buf[next_K_buf] + s * SLAB_BYTES;
                    const uint64_t a_desc = make_smem_desc(q_slab_addr);
                    const uint64_t b_desc = make_smem_desc(k_slab_addr);
                    const int enable_d = (s != 0) ? 1 : 0;
                    tcgen05_mma_f8f6f4(tmem_addr_n, a_desc, b_desc, IDESC_VAL, enable_d);
                }
                tcgen05_commit_mbar(mbar_s_buf[next_tmem]);
            }
        }

        // STAGE C: wait MMA[curr], read TMEM region (curr_tmem == pidx)
        mbarrier_wait_phase(mbar_s_buf[curr_tmem], phase_buf[curr_tmem]);
        phase_buf[curr_tmem] ^= 1;
        tcgen05_fence_after_sync();

        const uint32_t tmem_curr = tmem_col_buf[curr_tmem];
        const uint32_t taddr_lo  = (uint32_t(warp_id) * 32u) << 16 | tmem_curr;
        const uint32_t taddr_hi  = (uint32_t(warp_id) * 32u) << 16 | (tmem_curr + 32u);
        uint32_t rlo[32], rhi[32];
        tcgen05_ld_32x32b_x32(taddr_lo, rlo);
        tcgen05_ld_32x32b_x32(taddr_hi, rhi);
        tcgen05_wait_ld();
        // P8 FIX: order this warp's ld with subsequent MMA dispatches
        // (warp 0's MMA[next+1] in next iter could otherwise overwrite this region
        // before warp 2/3's ld fully completes — observed empirically as 16-pos
        // corruption pattern in HI half of warps 2 & 3).
        tcgen05_fence_before_sync();

        // STAGE D-1: scale apply + scores_smem
        const float* Sc = scale_smem_buf[pidx & 1];
        if (lane_id < 16) {
            const int row = warp_id * 16 + lane_id;
            float* row_out = scores_smem + row * SCORES_STRIDE;
            #pragma unroll
            for (int c = 0; c < 32; ++c) {
                float v = __uint_as_float(rlo[c]);
                row_out[c] = v * Sc[c];
            }
            #pragma unroll
            for (int c = 0; c < 32; ++c) {
                float v = __uint_as_float(rhi[c]);
                row_out[c + 32] = v * Sc[c + 32];
            }
        }
        __syncthreads();

        // STAGE D-2: H-reduce + GMEM write final_scores[blockIdx.y*256 + pidx*64 + col]
        if (tid < PAGE_SIZE) {
            const int col = tid;
            const int n_global = blockIdx.y * 256 + pidx * PAGE_SIZE + col;
            float acc0=0, acc1=0, acc2=0, acc3=0, acc4=0, acc5=0, acc6=0, acc7=0;
            #pragma unroll
            for (int h = 0; h < H; h += 8) {
                float s0 = scores_smem[(h  ) * SCORES_STRIDE + col];
                float s1 = scores_smem[(h+1) * SCORES_STRIDE + col];
                float s2 = scores_smem[(h+2) * SCORES_STRIDE + col];
                float s3 = scores_smem[(h+3) * SCORES_STRIDE + col];
                float s4 = scores_smem[(h+4) * SCORES_STRIDE + col];
                float s5 = scores_smem[(h+5) * SCORES_STRIDE + col];
                float s6 = scores_smem[(h+6) * SCORES_STRIDE + col];
                float s7 = scores_smem[(h+7) * SCORES_STRIDE + col];
                acc0 = fmaf(fmaxf(s0, 0.0f), weights_smem[h  ], acc0);
                acc1 = fmaf(fmaxf(s1, 0.0f), weights_smem[h+1], acc1);
                acc2 = fmaf(fmaxf(s2, 0.0f), weights_smem[h+2], acc2);
                acc3 = fmaf(fmaxf(s3, 0.0f), weights_smem[h+3], acc3);
                acc4 = fmaf(fmaxf(s4, 0.0f), weights_smem[h+4], acc4);
                acc5 = fmaf(fmaxf(s5, 0.0f), weights_smem[h+5], acc5);
                acc6 = fmaf(fmaxf(s6, 0.0f), weights_smem[h+6], acc6);
                acc7 = fmaf(fmaxf(s7, 0.0f), weights_smem[h+7], acc7);
            }
            float acc = ((acc0+acc1) + (acc2+acc3)) + ((acc4+acc5) + (acc6+acc7));
            final_scores[n_global] = acc;
        }

        // STAGE B: load K[later_pidx] into K_smem[curr_tmem] + write scale[later]
        if (has_later) {
            const int later_K_buf = later_pidx & 1;
            const uint8_t* k_l_src = k_pages_fp8 + later_pidx * 8192;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int u   = tid * 4 + i;
                int off = u << 4;
                *reinterpret_cast<uint4*>(K_smem_buf[later_K_buf] + off) =
                    *reinterpret_cast<const uint4*>(k_l_src + off);
            }
            if (tid < PAGE_SIZE) scale_smem_buf[later_K_buf][tid] = token_scale;
        }

        // STAGE E: ensure stage-B store visible before next iter's MMA dispatch
        if (has_later) __syncthreads();
    }

    __syncthreads();

    if (warp_id == 1) tcgen05_dealloc_256(tmem_col_base);
}

// ===== Host launcher =====
// P6 mode: B=1 → single CTA (matches earlier P6 baseline test).
// P7 mode: B>1 → grid (1, B) → many CTAs in parallel sharing same Q/K_pages.
void run_p6(
    torch::Tensor q_fp8,           // uint8 [8192]
    torch::Tensor k_pages_fp8,     // uint8 [4 * 8192 = 32768]
    torch::Tensor weights,         // float32 [64]
    double token_scale,            // applied uniformly
    torch::Tensor final_scores,    // float32 [B * 256]
    int64_t B)                     // num parallel CTAs
{
    const at::cuda::CUDAGuard device_guard(q_fp8.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    dim3 grid(1, static_cast<int>(B));
    tmem2_p6_kernel<<<grid, 128, SMEM_BYTES, stream>>>(
        q_fp8.data_ptr<uint8_t>(),
        k_pages_fp8.data_ptr<uint8_t>(),
        weights.data_ptr<float>(),
        static_cast<float>(token_scale),
        final_scores.data_ptr<float>());
    AT_CUDA_CHECK(cudaGetLastError());
}
