// dsa_topk_indexer_fp8_h64_d128_topk2048_ps64
// Tier 3 + Option C: fused tcgen05 FP8 MMA + SMEM-transpose H-reduction.
// Per-page: FP8 K HBM -> SMEM 8xT -> tcgen05.mma.kind::f8f6f4 -> TMEM -> tcgen05.ld
// -> scale multiply -> scores_smem[64×64] -> 64 threads each read DOWN a column with
// a 64-FMA weighted-relu-sum -> final_scores[B, N_max] GMEM.
// Eliminates the scores_all[B,64,N_max] round trip without the shuffle-tree critical
// path that caused GT-29's Tier 4(b) regression. Downstream (at::topk, translate) unchanged.

#include <cstdint>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

constexpr int INDEX_HEAD_DIM  = 128;
constexpr int PAGE_SIZE       = 64;
constexpr int PAGE_DATA_BYTES = 8192;
constexpr int PAGE_BYTES      = 8448;
constexpr int TOPK            = 2048;
constexpr int H               = 64;       // num query heads

// Tier 3 constants
constexpr int PAGES_PER_CTA   = 4;       // GT-13 min=2
constexpr int MMA_M           = 64;      // matches H exactly
constexpr int MMA_N           = 64;      // one page per N-tile (no straddling)
constexpr int MMA_K           = 32;      // f8f6f4 fixed per Table 41
constexpr int NUM_SLABS       = INDEX_HEAD_DIM / MMA_K;  // 4
// IDESC: M=64, N=64, K=32, dtype=F32(=1), atype=E4M3(=0), btype=E4M3(=0),
// no transpose/negate/sparsity/.ws. See d3.md Decision 4.
constexpr uint32_t IDESC_VAL  = (1u << 4) | (8u << 17) | (4u << 24);  // 0x04100010

// SMEM layout (all 16-B aligned; see d4.md address composition)
constexpr int Q_SMEM_OFFSET       = 0;                   // 8192 B
constexpr int K_SMEM_OFFSET       = 8192;                // 8192 B
constexpr int SCORES_SMEM_OFFSET  = 16384;               // 16384 B ([64 rows × 64 cols] FP32)
constexpr int SCALE_SMEM_OFFSET   = 32768;               // 256 B
constexpr int BT_CACHE_OFFSET     = 33024;               // 16 B
constexpr int ALLOC_SLOT_OFFSET   = 33040;               // 4 B
constexpr int MBAR_SLOT_OFFSET    = 33056;               // 8 B
constexpr int WEIGHTS_SMEM_OFFSET = 33072;               // 256 B ([H=64] FP32)  — Option C
constexpr int SMEM_BYTES          = 33328;

// GT-11 8xT core-tile constants
constexpr int SLAB_BYTES = 2048;
constexpr int SBO_BYTES  = 256;
constexpr int LBO_BYTES  = 128;

// 8xT fill formula: for an FP8 byte at logical (m, k) in a [M_tile, K_tile=128] operand
// stored across 4 K-slabs, the SMEM byte offset is:
__device__ __forceinline__ int smem_8xT_offset(int m, int k) {
    return (k / 32) * SLAB_BYTES
         + (m / 8)  * SBO_BYTES
         + ((k % 32) / 16) * LBO_BYTES
         + (m % 8)  * 16
         + (k % 16);
}

// SMEM descriptor encode helper (Table 42): 14-bit field units of 16 B
__device__ __forceinline__ uint64_t desc_encode_u64(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}

// Build MMA SMEM descriptor for a K-major 8xT operand at the given shared-space
// address (lower 24 bits). SBO=256, LBO=128 (GT-11). Swizzle=None (GT-10).
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t smem_addr_shared) {
    uint64_t d = 0;
    d |= desc_encode_u64((uint64_t)smem_addr_shared);   // bits 0-13 matrix start
    d |= desc_encode_u64((uint64_t)LBO_BYTES) << 16;    // bits 16-29 LBO -> 8
    d |= desc_encode_u64((uint64_t)SBO_BYTES) << 32;    // bits 32-45 SBO -> 16
    d |= (uint64_t)0b001ULL << 46;                      // bits 46-48 fixed
    // bits 49+ all 0: base offset, stride mode, swizzle = NONE
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

// GT-18: commit form for CUDA 12.8 on sm_100a
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

// FP8 MMA. A and B are both K-major 8xT in SMEM, transpose_A=transpose_B=0 (GT-15).
// enable_input_d = 0 -> fresh accumulator (D = A*B); 1 -> accumulate (D = A*B + D).
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

// tcgen05.ld.32x32b.x32 — loads 32 cols of 32 TMEM lanes into 32 FP32 regs per thread.
// Per warp thread L: TMEM lane (warp_id*32 + L), 32 consecutive columns from taddr.
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

// =====================================================================
//  Fused FP8 MMA + H-reduction kernel (Tier 3 + Option C)
//  Grid:  (ceil(max_num_pages / PAGES_PER_CTA), B)
//  Block: 128 threads (4 warps)
//  SMEM:  ~33 KiB
//  Output: final_scores[B, N_max] (H-reduction done inline; no scores_all intermediate)
// =====================================================================
__global__ __launch_bounds__(128, 2)
void fp8_mma_final_scores_kernel(
    const uint8_t* __restrict__ q_fp8,          // [B, 64, 128] FP8 (as raw bytes)
    const uint8_t* __restrict__ k_cache,        // paged FP8 cache (uint8 view)
    const float*   __restrict__ weights,        // [B, 64] FP32
    const int32_t* __restrict__ block_table,    // [B, max_num_pages]
    const int32_t* __restrict__ seq_lens,       // [B]
    float*         __restrict__ final_scores,   // [B, N_max] FP32
    int max_num_pages,
    int N_max)
{
    extern __shared__ __align__(16) uint8_t smem_raw[];

    uint8_t* Q_smem          = smem_raw + Q_SMEM_OFFSET;
    uint8_t* K_smem          = smem_raw + K_SMEM_OFFSET;
    float*   scores_smem     = reinterpret_cast<float*>(smem_raw + SCORES_SMEM_OFFSET);
    float*   scale_smem      = reinterpret_cast<float*>(smem_raw + SCALE_SMEM_OFFSET);
    int32_t* bt_cache_smem   = reinterpret_cast<int32_t*>(smem_raw + BT_CACHE_OFFSET);
    uint32_t* alloc_slot_ptr = reinterpret_cast<uint32_t*>(smem_raw + ALLOC_SLOT_OFFSET);
    float*   weights_smem    = reinterpret_cast<float*>(smem_raw + WEIGHTS_SMEM_OFFSET);
    // mbar_slot at MBAR_SLOT_OFFSET (accessed by shared-space addr, not via C++ ptr)

    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;
    const int lane_id   = tid & 31;
    const int b         = blockIdx.y;
    const int cta_page_start = blockIdx.x * PAGES_PER_CTA;

    // ---- P0: kernel-scope setup (thread-private regs) ----
    const int seq_len_b = seq_lens[b];

    // Shared-space (24-bit) addresses for descriptors / asm
    const uint32_t q_shared       = __cvta_generic_to_shared(Q_smem);
    const uint32_t k_shared       = __cvta_generic_to_shared(K_smem);
    const uint32_t alloc_slot_s   = __cvta_generic_to_shared(alloc_slot_ptr);
    const uint32_t mbar_s         = __cvta_generic_to_shared(smem_raw + MBAR_SLOT_OFFSET);

    // ---- P1: TMEM alloc (warp 1) ----
    if (warp_id == 1) {
        tcgen05_alloc_64(alloc_slot_s);
    }
    __syncthreads();
    const uint32_t tmem_col = alloc_slot_ptr[0];   // column base returned by alloc

    // ---- P2: block_table cache (warp 0, thread 0) ----
    if (warp_id == 0 && lane_id == 0) {
        #pragma unroll
        for (int p = 0; p < PAGES_PER_CTA; ++p) {
            const int pg_id = cta_page_start + p;
            bt_cache_smem[p] = (pg_id < max_num_pages)
                ? block_table[b * max_num_pages + pg_id] : 0;
        }
    }

    // ---- P3: mbarrier init (warp 0, thread 0) ----
    if (warp_id == 0 && lane_id == 0) {
        mbarrier_init_1(mbar_s);
    }
    // ---- P3b: weights cooperative load (parallel with mbar init) — Option C ----
    if (tid < H) {
        weights_smem[tid] = weights[b * H + tid];
    }
    uint32_t phase = 0;
    __syncthreads();

    // ---- P4: Q cooperative load -> Q_smem (8xT) ----
    // 128 threads * 4 uint4 = 512 uint4 = 8192 B.
    {
        const uint8_t* q_base = q_fp8 + size_t(b) * 64 * INDEX_HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int u        = tid * 4 + i;
            int m        = u >> 3;        // u / 8  (m ∈ [0, 64))
            int k_uint4  = u & 7;         // u % 8  (∈ [0, 8))
            int k_base   = k_uint4 << 4;  // * 16
            uint4 val = *reinterpret_cast<const uint4*>(q_base + m * INDEX_HEAD_DIM + k_base);
            int off = smem_8xT_offset(m, k_base);  // (k_base % 16) == 0 -> uint4 boundary
            *reinterpret_cast<uint4*>(Q_smem + off) = val;
        }
    }
    __syncthreads();

    // ---- Page loop ----
    const uint4 zero_uint4 = make_uint4(0u, 0u, 0u, 0u);

    for (int pidx = 0; pidx < PAGES_PER_CTA; ++pidx) {
        const int page_id = cta_page_start + pidx;
        if (page_id >= max_num_pages) break;    // falls through to dealloc

        const int32_t phys    = bt_cache_smem[pidx];
        const size_t  pg_base = size_t(phys) * PAGE_BYTES;

        // ---- P6: K cooperative load with zero-fill mask ----
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int u         = tid * 4 + i;
            int tok       = u >> 3;        // ∈ [0, 64)
            int k_uint4   = u & 7;
            int k_base    = k_uint4 << 4;
            int global_t  = page_id * PAGE_SIZE + tok;
            int off       = smem_8xT_offset(tok, k_base);
            if (global_t >= seq_len_b) {
                *reinterpret_cast<uint4*>(K_smem + off) = zero_uint4;
            } else {
                uint4 val = *reinterpret_cast<const uint4*>(
                    k_cache + pg_base + tok * INDEX_HEAD_DIM + k_base);
                *reinterpret_cast<uint4*>(K_smem + off) = val;
            }
        }

        // ---- P7: scale cooperative load with zero-fill ----
        if (tid < PAGE_SIZE) {
            int global_t = page_id * PAGE_SIZE + tid;
            if (global_t >= seq_len_b) {
                scale_smem[tid] = 0.0f;
            } else {
                scale_smem[tid] = *reinterpret_cast<const float*>(
                    k_cache + pg_base + PAGE_DATA_BYTES + tid * 4);
            }
        }
        __syncthreads();

        // ---- P8: MMA issue (4 slabs, NO unroll per GT-14) ----
        // ---- P9: commit (inside warp 0 elected) ----
        if (warp_id == 0 && elect_one_sync()) {
            const uint32_t tmem_addr = tmem_col;
            #pragma unroll 1
            for (int s = 0; s < NUM_SLABS; ++s) {
                const uint32_t q_slab_addr = q_shared + s * SLAB_BYTES;
                const uint32_t k_slab_addr = k_shared + s * SLAB_BYTES;
                const uint64_t a_desc = make_smem_desc(q_slab_addr);
                const uint64_t b_desc = make_smem_desc(k_slab_addr);
                const int enable_d = (s != 0) ? 1 : 0;
                tcgen05_mma_f8f6f4(tmem_addr, a_desc, b_desc, IDESC_VAL, enable_d);
            }
            tcgen05_commit_mbar(mbar_s);
        }

        // ---- P10: mbarrier wait (all threads) ----
        mbarrier_wait_phase(mbar_s, phase);
        phase ^= 1;

        // ---- P11: fence (all threads) ----
        tcgen05_fence_after_sync();

        // ---- P12: tcgen05.ld (all 32 lanes per warp — GT-17) ----
        // Load full [32 lanes x 64 cols] accumulator for this warp via 2 x .32x32b.x32.
        const uint32_t taddr_lo = (uint32_t(warp_id) * 32u) << 16 | tmem_col;
        const uint32_t taddr_hi = (uint32_t(warp_id) * 32u) << 16 | (tmem_col + 32u);
        uint32_t rlo[32], rhi[32];
        tcgen05_ld_32x32b_x32(taddr_lo, rlo);
        tcgen05_ld_32x32b_x32(taddr_hi, rhi);

        // ---- P13: wait::ld (all threads) ----
        tcgen05_wait_ld();

        // ---- P14: scale apply + masked write to scores_smem ----
        // GT-17: only lanes 0..15 carry live M-rows; lanes 16..31 read zeros — skip them.
        if (lane_id < 16) {
            const int row = warp_id * 16 + lane_id;
            float* row_out = scores_smem + row * MMA_N;
            #pragma unroll
            for (int c = 0; c < 32; ++c) {
                float v = __uint_as_float(rlo[c]);
                row_out[c]      = v * scale_smem[c];
            }
            #pragma unroll
            for (int c = 0; c < 32; ++c) {
                float v = __uint_as_float(rhi[c]);
                row_out[c + 32] = v * scale_smem[c + 32];
            }
        }
        __syncthreads();

        // ---- P15 (Option C): SMEM-transpose H-reduction + GMEM write final_scores ----
        // 64 threads (one per N column) read DOWN the column across all 64 H-rows,
        // fusing scale*score (already in SMEM) -> relu -> weight -> sum-over-H.
        // No shuffle-tree critical path (GT-29); just a 64-iter FMA chain per thread.
        // Bank layout: 64 threads × stride-64 reads = 2-way SMEM bank conflict, accepted.
        if (tid < PAGE_SIZE) {
            const int col      = tid;
            const int n_global = page_id * PAGE_SIZE + col;
            float acc = 0.0f;
            #pragma unroll 16
            for (int h = 0; h < H; ++h) {
                float s = scores_smem[h * MMA_N + col];
                acc = fmaf(fmaxf(s, 0.0f), weights_smem[h], acc);
            }
            final_scores[b * N_max + n_global] =
                (n_global >= seq_len_b) ? -INFINITY : acc;
        }
        __syncthreads();
    }

    // ---- P16: TMEM dealloc (single exit path) ----
    if (warp_id == 1) {
        tcgen05_dealloc_64(tmem_col);
    }
}

// =====================================================================
//  Translate indices (Tier 2, unchanged)
// =====================================================================
__global__ void translate_indices_batched_kernel(
    const int64_t* __restrict__ local_idx_all,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    int32_t*       __restrict__ topk_indices,
    int max_num_pages,
    int k_req)
{
    const int b = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= k_req) return;

    const int sl = seq_lens[b];
    const int actual_topk = sl < TOPK ? sl : TOPK;
    if (t >= actual_topk) return;

    const int64_t lid  = local_idx_all[size_t(b) * k_req + t];
    const int32_t phys = block_table[b * max_num_pages + static_cast<int>(lid >> 6)];
    topk_indices[b * TOPK + t] = phys * PAGE_SIZE + static_cast<int32_t>(lid & 63);
}

// =====================================================================
//  Host entry point (unchanged signature; config.toml entry_point unchanged)
// =====================================================================
void launch_topk_c(
    torch::Tensor q_index_fp8,
    torch::Tensor k_index_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    torch::Tensor topk_indices)
{
    const at::cuda::CUDAGuard device_guard(q_index_fp8.device());
    const int B             = static_cast<int>(q_index_fp8.size(0));
    const int max_num_pages = static_cast<int>(block_table.size(1));
    const int N_max         = max_num_pages * PAGE_SIZE;
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // View Q as raw bytes (FP8 e4m3 interpreted as uint8 for MMA consumption).
    auto q_u8      = q_index_fp8.view(torch::kUInt8);
    auto k_cache_u8 = k_index_cache_fp8.view(torch::kUInt8);

    const uint8_t* q_ptr       = q_u8.data_ptr<uint8_t>();
    const uint8_t* k_cache_ptr = k_cache_u8.data_ptr<uint8_t>();
    const float*   w_ptr       = weights.data_ptr<float>();
    const int32_t* bt_ptr      = block_table.data_ptr<int32_t>();
    const int32_t* sl_ptr      = seq_lens.data_ptr<int32_t>();

    auto opts_f32 = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(q_index_fp8.device());

    // Phases 1+2+3 (fused via Option C): MMA + scale + relu + weights + sum-over-heads.
    auto final_scores = torch::empty({B, N_max}, opts_f32);

    const int grid_x = (max_num_pages + PAGES_PER_CTA - 1) / PAGES_PER_CTA;
    dim3 grid(grid_x, B);
    fp8_mma_final_scores_kernel<<<grid, 128, SMEM_BYTES, stream>>>(
        q_ptr, k_cache_ptr, w_ptr, bt_ptr, sl_ptr,
        final_scores.data_ptr<float>(),
        max_num_pages, N_max);
    AT_CUDA_CHECK(cudaGetLastError());

    // Phase 4: single batched top-K along dim -1.
    const int k_req = std::min(TOPK, N_max);
    auto result = at::topk(final_scores, k_req, -1, true, true);
    auto local_idx = std::get<1>(result).contiguous();

    // Phase 5: batched index translation.
    topk_indices.fill_(-1);
    translate_indices_batched_kernel<<<
        dim3((k_req + 255) / 256, B), 256, 0, stream>>>(
        local_idx.data_ptr<int64_t>(),
        bt_ptr, sl_ptr,
        topk_indices.data_ptr<int32_t>(),
        max_num_pages,
        k_req);
    AT_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_topk_c", &launch_topk_c, "DSA TopK FP8 indexer (Tier 3 + Option C)");
}
