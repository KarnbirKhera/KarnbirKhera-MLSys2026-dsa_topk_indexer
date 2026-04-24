# DSA TopK FP8 Kernel — Generation and Optimization Workflow
# Target: NVIDIA B200 (sm_100a) via Modal

---

## Section 1 — Kernel Identity

**Definition name:** `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`

**What it computes:** For each batch element, compute sparse attention scores over a paged FP8 KV cache using 64 indexer heads, select the top-2048 token indices by score, and return global token indices. Formula: `final_scores = sum(relu(q @ K.T) * weights)` across heads, then `topk(final_scores, 2048)`.

**Fixed constants — never re-derive these:**

| Constant | Value | Meaning |
|---|---|---|
| `INDEX_HEAD_DIM` | 128 | Head dimension |
| `PAGE_SIZE` | 64 | Tokens per KV cache page |
| `TOPK` | 2048 | Number of top-K indices to select |
| `PAGE_DATA_BYTES` | 8192 | FP8 data bytes per page (64 x 128) |
| `PAGE_BYTES` | 8448 | Total bytes per page (64 x 132) |

**FP8 memory layout — the single hardest correctness landmine:**
Each page is packed as `[page_size x 128 FP8 bytes][page_size x 4 scale bytes]` — all FP8 data first, then all scales. The tensor arrives shaped `[num_pages, page_size, 1, 132]` with dtype `int8`, but must be interpreted as `uint8`. Scale for token `tok` on page `phys` is at byte offset `PAGE_DATA_BYTES + tok * 4` from the page base. Do NOT attempt per-token `[fp8, scale]` indexing directly from the 132-wide dimension — the layout is not interleaved.

**Variable axes (differ per workload):** `batch_size`, `max_num_pages`, `num_pages`

**Output:** `topk_indices [batch_size, 2048]` dtype int32. Positions beyond `actual_topk = min(TOPK, seq_len)` are filled with `-1`.

---

## Section 2 — GT Update Rule STAR MOST IMPORTANT RULE IN THIS FILE STAR

Every time a test run confirms a new hardware constraint, immediately write a new GT-N entry at the bottom of Section 7 (always-active) or Section 11 (Tier 3 only), depending on scope. Include the date and the probe or test that confirmed it.

**This rule is non-negotiable and applies after every diagnostic resolution.** It is what prevents the same failure from occurring twice. A diagnosis that does not produce a GT entry is incomplete.

Format:
```
### GT-N: [short title]
[What was confirmed, what the correct behavior is, what the wrong behavior looks like.]
Confirmed YYYY-MM-DD on Modal B200 via [probe name or test that confirmed it].
```

---

## Section 3 — Project Structure

```
~/CUDAExperiments/ClaudeCode3.0/
├── CLAUDE.md                                      <- this file
├── fi-bench-env/                                  <- Python venv (activate before any command)
├── mlsys26-contest/                               <- dataset (3GB, LFS blobs present)
│   ├── definitions/dsa_paged/dsa_paged/           <- NOTE: double dsa_paged — not a typo
│   │   └── dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json   <- Python reference + spec
│   ├── workloads/dsa_paged/
│   │   └── dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl  <- 128 workloads
│   └── blob/workloads/dsa_paged/
│       └── dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/       <- 128 safetensors confirmed present
├── flashinfer-bench-starter-kit/
│   ├── config.toml                                <- definition + entry_point (keep in sync)
│   ├── solution/cuda/
│   │   └── kernel.cu                              <- THE file the agent edits
│   └── scripts/
│       ├── run_modal.py                           <- test command
│       └── pack_solution.py                       <- packs kernel.cu into solution JSON
├── checkpoints/
│   └── kernel_naive.cu                            <- LOCKED 2026-04-19. NEVER modified.
├── framework/                                     <- derivation framework (Phase 2 Tier 2+)
├── ptx_isa_sections/                              <- PTX ISA reference (Phase 2 Tier 3 only)
└── gau-nernst_reference.h                         <- B200 PTX wrappers (Phase 2 Tier 3 only)
```

**Definition JSON path — write it literally, do not construct from op_type:**
`mlsys26-contest/definitions/dsa_paged/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json`

**Python reference ground truth:** The `reference` field inside the definition JSON contains the canonical `run()` function. This is the authoritative specification for what the kernel must compute. On any correctness failure, diff the kernel output against this reference on the smallest failing workload.

---

## Section 4 — Environment

Before running any Python command or test:

```bash
source ~/CUDAExperiments/ClaudeCode3.0/fi-bench-env/bin/activate
```

If this is not active, all `modal` and `flashinfer_bench` commands will fail silently or use the wrong packages. Verify with `which python` — it must show the venv path.

**Modal workspace:** `khera-karnbir`
**Modal volume:** `flashinfer-trace` — contains the dataset at `/data/mlsys26-contest`
**FIB_DATASET_PATH** (for local reference only): `~/CUDAExperiments/ClaudeCode3.0/mlsys26-contest`

All benchmarking runs on Modal B200. Never attempt to run benchmarks locally — the local GPU is an RTX 4060 (sm_89) and will produce RUNTIME_ERROR on sm_100a targeted binaries.

---

## Section 5 — The Two Phases

```
Phase 1: COMPLETE. checkpoint locked at checkpoints/kernel_naive.cu 2026-04-19.
Phase 2: Per-optimization cycle (tiered) -> D1-V4 scoped to delta -> implement -> test -> keep/revert
```

Phase 1 is complete. `checkpoints/kernel_naive.cu` is locked. Phase 2 begins now.

Phase 2 uses the derivation framework — but scoped per optimization and tiered by complexity. D1-V4 fires once per optimization change, not once at the start of the entire phase.

---

## Section 6 — Phase 1: COMPLETE

Phase 1 passed 128/128 on 2026-04-19. Checkpoint locked.

**Baseline performance (Modal B200, confirmed 2026-04-19):**

| Metric | Value |
|---|---|
| Pass rate | 128/128 |
| Correctness | abs_err=0.00, rel_err=0.00 on all workloads |
| Mean latency | 0.803 ms |
| p50 latency | 0.763 ms |
| p95 latency | 1.587 ms |
| Min latency | 0.111 ms (batch=1, max_num_pages=1) |
| Max latency | 1.656 ms (large batch, max_num_pages=90+) |
| Speedup vs Python reference | 3.4x–8.8x |

Every Phase 2 optimization is measured against this baseline. Any optimization that does not improve mean latency AND passes 128/128 is not worth keeping.

**Revert command (restore Phase 1 state at any time):**
```bash
cp ~/CUDAExperiments/ClaudeCode3.0/checkpoints/kernel_naive.cu \
   ~/CUDAExperiments/ClaudeCode3.0/flashinfer-bench-starter-kit/solution/cuda/kernel.cu
```

### Phase 1 audit checklist (archived — verified passing)

1. FP8 layout: k_cache viewed as uint8 before byte arithmetic. CONFIRMED.
2. Scale address: pb + PAGE_DATA_BYTES + tok * 4. CONFIRMED.
3. block_table indexing: global token = phys * PAGE_SIZE + offset. CONFIRMED.
4. actual_topk = min(TOPK, seq_len) applied before at::topk. CONFIRMED.
5. Sentinel fill: topk_indices.fill_(-1) before batch loop. CONFIRMED.
6. Output shape: [batch_size, 2048] dtype int32. CONFIRMED.
7. Short-circuit: seq_len <= 0 handled with continue. CONFIRMED.
8. 128/128 PASSED on Modal B200. CONFIRMED.

---

## Section 7 — Always-Active GT Entries

These apply in both Phase 1 and Phase 2. Read these before any implementation work.

### GT-8: Roofline — Memory Bound, Pipeline Absent

DSA TopK: ~187 FLOPs/byte. B200 ridge point: ~495 FLOPs/byte. This kernel is memory-bound. Pipelining (async loads to hide latency) is ABSENT — the compute intensity does not justify it. Do not add pipeline infrastructure. Do not add cp.async prefetch stages. The bottleneck is HBM bandwidth, not compute.

### GT-9: TMA Stride Constraint — K-Cache Cannot Use TMA

`cuTensorMapEncodeTiled` requires all globalStrides to be multiples of 16 bytes. The K-cache has 132-byte rows (128 FP8 data + 4 scale bytes). 132 is NOT a multiple of 16. Using TMA for K-cache loads will produce a silently invalid tensor map that causes XID 13 at runtime with no useful error message. Use cooperative thread copy (regular ld.global) for K-cache loads. Confirmed 2026-04-12 on Modal B200 via bisection.

### GT-13: PAGES_PER_CTA Minimum = 2

PAGES_PER_CTA=1 produces non-deterministic results on B200. The minimum confirmed stable value is 2. Any kernel restructuring that introduces a PAGES_PER_CTA parameter must set its minimum to 2. Confirmed 2026-04-15 on Modal B200.

### GT-19: Torch Binding Requires nvidia/cuda devel Base Image on Modal

The flashinfer-bench TorchBuilder uses `torch.utils.cpp_extension.load()` to JIT-compile CUDA kernels. This requires `nvcc`, CUDA headers (including `cuda_fp8.h`), and build tools to be present in the Modal container. The default `debian_slim` base image does NOT include these. The correct Modal image uses `nvidia/cuda:12.8.1-devel-ubuntu22.04` as the base. Using any non-devel CUDA image produces COMPILE_ERROR on all workloads with no error message visible in the local terminal.

Additionally, `flashinfer-bench` must be installed from git source (not pip) to match the evaluation environment:
```python
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12"
    )
    .apt_install("build-essential", "ninja-build", "git")
    .run_commands(
        "git clone https://github.com/flashinfer-ai/flashinfer-bench.git /flashinfer-bench "
        "&& cd /flashinfer-bench && pip install -v -e ."
    )
    .pip_install("torch", "triton", "numpy", "ninja")
)
```
Confirmed 2026-04-19 on Modal B200 — resolves all COMPILE_ERROR failures.

### GT-20: pack_solution.py Does Not Read binding Field Without Patch

The starter kit's `pack_solution.py` does not pass the `binding` field from `config.toml` to `BuildSpec`. Without patching, `binding` defaults to None (tvm-ffi), causing COMPILE_ERROR when the kernel uses torch binding. The fix — add two lines to `pack_solution.py` in the `pack_solution()` function:

```python
binding = build_config.get("binding", None)
spec = BuildSpec(
    language=language,
    target_hardware=["NVIDIA_B200"],
    entry_point=entry_point,
    destination_passing_style=dps,
    **({"binding": binding} if binding else {}),
)
```

Confirmed 2026-04-19 — without this patch, binding=torch kernels fail with COMPILE_ERROR despite correct kernel code.

### GT-BASELINE: Naive Kernel Performance Baseline

Locked 2026-04-19. `checkpoints/kernel_naive.cu` passes 128/128 on B200.
Mean: 0.803 ms | p50: 0.763 ms | p95: 1.587 ms | min: 0.111 ms | max: 1.656 ms
Speedup over Python reference: 3.4x–8.8x depending on workload size.
Every optimization in Phase 2 is measured against this baseline.

### GT-26: Fully-Batched Launch Path — Tier 2 Win (−86% mean)

Collapsing the per-batch-element host loop into a single set of batched GPU launches yields a dominant Tier 2 win. Confirmed run on Modal B200 (app ap-4Q3p4iX7F64dgppG3tvZIO): mean 0.803→0.112 ms (−86.1% vs GT-BASELINE; −82.2% vs Tier 1 best 0.630 ms), p50 0.763→0.099 ms (−87.0%), p95 1.587→0.246 ms (−84.5%), min 0.111→0.048 ms (−56.8%), max 1.656→0.259 ms (−84.4%). 128/128 PASS, abs_err=0 rel_err=0, per-workload speedup vs Python reference 20–89×.

Structure (replaces the B-iteration host loop entirely):
1. Batched dequant: grid `(max_num_pages, PAGE_SIZE, B)`, block 32; padding rows (row ≥ `seq_lens[b]`) zero-filled so `bmm` at those columns contributes 0 to scores.
2. `torch::bmm(q_fp32_all, K_deq_all.transpose(-2, -1))` — one cuBLAS batched call replaces B per-element matmuls.
3. Batched fused `weighted_relu_sum`: grid `((N_max+255)/256, B)`, block 256. Stamps −INF at positions `t ≥ seq_lens[b]` so padded columns sort to the bottom of `at::topk`.
4. Single `at::topk(final_scores, k_req, -1, true, true)` with `k_req = min(TOPK, N_max)` — one batched radix-select.
5. Batched `translate_indices`: handles the `local_idx` stride correctly as `k_req` (not `TOPK` — the bug that initially produced 125 RUNTIME_ERRORs on max_num_pages < 32 workloads), leaving positions `[actual_topk_b, TOPK)` as the −1 sentinel from `topk_indices.fill_(-1)`.

Why the previous Tier 1 analysis underestimated this: the Tier 1 dominant cost hypothesis was "HBM traffic on K" (dequant write + matmul read). The real dominant cost was CPU-side kernel-launch enqueueing in the batch loop — the loop issued ~5 kernels per batch element, and for batches up to 32 that's 160 launches queued sequentially. Batching collapses that to 5 launches total per invocation. Bonus: seq_lens stays on GPU throughout — implicitly neutralizing the cost GT-22 measured.

GT-23's 32-thread-per-block rule for dequant is preserved (the batched version still uses 32 threads, one warp per `(page, tok, batch)` triple). Confirmed 2026-04-19 on Modal B200.

### GT-25: Pre-allocating final_scores Scratch Outside Batch Loop — Micro-Win

Moving `torch::empty({sl}, opts)` for the fused kernel's output out of the batch loop (allocate once per call at `{max_num_pages*PAGE_SIZE}`, `.slice(0,0,sl)` per iteration) gives a further small win on top of GT-24. Confirmed run on Modal B200 (app ap-pXQnjixeAROZ8MguZPG3vM): mean 0.639→0.630 ms (−1.4%), p50 0.598→0.590 ms (−1.3%), p95 1.299→1.286 ms (−1.0%), min 0.082→0.081 ms, max 1.450→1.438 ms. 128/128 PASS. Mechanism: saves B `torch::empty` calls per launch; torch caching allocator is cheap but not free. Note: this directly contradicts the spirit of GT-21 only because the alternative here is still torch-managed memory — we are not replacing torch's allocator, just hoisting a repeat call. Confirmed 2026-04-18 on Modal B200.

### GT-TIER-SUMMARY: Cumulative State After Tier 1 + Tier 2 (2026-04-19)

Against the locked GT-BASELINE (`checkpoints/kernel_naive.cu`, 0.803 ms mean):
- mean 0.803 → **0.112 ms** (**−86.1%**)
- p50 0.763 → **0.099 ms** (−87.0%)
- p95 1.587 → **0.246 ms** (−84.5%)
- min 0.111 → **0.048 ms** (−56.8%)
- max 1.656 → **0.259 ms** (−84.4%)

Kept: GT-24 (fused weighted_relu_sum kernel), GT-25 (pre-allocated final_scores scratch, superseded inside the batched rewrite), GT-26 (fully batched launches — the dominant win).
Reverted with recorded failure GTs: GT-21 (static K_deq scratch), GT-22 (pinned-host seq_lens), GT-23 (wider dequant blocks).

Critical correction to the Tier 1 dominant-cost hypothesis: the bottleneck was CPU-side kernel-launch enqueueing in the host batch loop, NOT HBM traffic on K. Once the loop was removed (GT-26), the kernel is 5–8× faster than Tier 1 best. Remaining cost centers (for any future Tier 3 work): cuBLAS `bmm` on FP32 (still materializes K as FP32), `at::topk` on `[B, N_max]`, and the dequant FP32 materialization. If further speedups are needed, Tier 3 candidates: fused dequant+BF16-or-FP8 bmm via tcgen05 tensor cores (GT-2 family), or custom radix-select for the top-K on larger `N_max`. Not pursued unless target mean < 0.1 ms is required.

### GT-24: Fused weighted_relu_sum Kernel — First Tier 1 Win (−20.4% mean)

Replacing the torch chain `(torch::relu(scores) * weights[b].unsqueeze(-1)).sum(0)` with a single custom kernel `weighted_relu_sum_kernel` that reads `scores[H=64, N]` column-wise, applies relu and per-head weights in registers, reduces across H, and writes `final_scores[N]` once — is a clean Tier 1 win. Confirmed run on Modal B200 (app ap-9UnSSiakOZYpHdOly6D73W): mean 0.803→0.639 ms (−20.4%), p50 0.763→0.598 ms (−21.6%), p95 1.587→1.299 ms (−18.1%), min 0.111→0.082 ms (−26.1%), max 1.656→1.450 ms (−12.4%). 128/128 PASS, abs_err=0.00 rel_err=0.00. Mechanism: eliminates 2 kernel launches per batch element (relu, multiply; sum was its own kernel too — so really 3 → 1) and cuts HBM traffic on the `[64, N]` scores tensor from ~3× (relu reads+writes, multiply reads+writes, sum reads) to 1× (fused kernel reads once). Kernel config: 256 threads per block, grid = ceil(N/256); weights[0..63] loaded once per block into SMEM. Result is bit-identical to the torch chain because the computation is a strict fusion — no precision change. This is now the baseline for subsequent Tier 1 optimizations. Confirmed 2026-04-18 on Modal B200.

### GT-23: dequant_gather_kernel 32-Thread Blocks Beat Wider Blocks

The naive dequant_gather launch `<<<dim3(num_pages_seq, 64), 32>>>` (one warp per (page, token) pair) outperforms wider block tilings. Confirmed run on Modal B200: changing to `<<<dim3(num_pages_seq, 16), 128>>>` (4 tokens per block, 4 warps, per-warp arithmetic byte-identical) regresses significantly — mean 0.803→0.924 ms (+15.1%), p50 0.763→0.860 ms (+12.7%), p95 1.587→1.881 ms (+18.5%), max 1.656→1.974 ms (+19.2%), min unchanged at 0.111 ms. Root cause hypothesis: the dequant kernel is HBM-bandwidth-bound (128 FP8 bytes + 4 scale bytes in → 512 FP32 bytes out per token). With 32-thread blocks, many small blocks give the warp scheduler fine-grained slots across the 148 SMs and hide tail effects; with 128-thread blocks, each block is a heavier scheduling unit that still cannot accelerate the memory-bound work, and the hit shows up most on the largest workloads (p95/max +18-19%) where memory contention dominates. Do not widen dequant_gather blocks — the naive 32-thread-per-block shape is correct for this memory-bound pattern. If a fused dequant+matmul kernel is considered (Tier 2/3), that is a different design and this constraint does not carry over. Confirmed 2026-04-18 on Modal B200 (Modal app ap-D0WYFAfi2HCgIpO1CvQ80X).

### GT-22: Do Not Replace torch Host-Side Primitives With Explicit cudaRuntime Calls

Replacing `seq_lens.to(torch::kCPU)` with an explicit `cudaMemcpyAsync` into a persistent pinned-host buffer + `cudaStreamSynchronize(stream)` is a REGRESSION. Confirmed run on Modal B200: mean 0.803→0.830 ms (+3.4%), p50 0.763→0.792 ms (+3.8%), p95 1.587→1.728 ms (+8.9%), max 1.656→1.814 ms (+9.5%). Only min improved (0.111→0.102 ms — the smallest batch where pinned setup amortizes well). Root cause: torch's `.to(kCPU)` uses lower-overhead primitives (targeted event waits, internal caching for destination CPU allocation, tighter driver paths) than an application-level `cudaMemcpyAsync`/`cudaStreamSynchronize` pair. For tiny D2H transfers (≤128 bytes) on the same CUDA stream, the driver API overhead of explicit calls exceeds any overlap benefit, and since `fill_/to_f32` serialize on the same stream, the "overlap" with prior work never materializes. Combined with GT-21: do NOT attempt to replace torch host-side primitives (allocator, D2H/H2D copy, synchronization) with explicit cudaRuntime equivalents for tiny per-launch operations. The wins must come from GPU-side kernel changes, not host-side substitution. Confirmed 2026-04-18 on Modal B200 (Modal app ap-KGVC2B6lkSvoTKDcVjAZhw).

### GT-21: torch Caching Allocator Beats Static cudaMalloc for K_deq Scratch

Replacing `torch::empty({max_num_pages*PAGE_SIZE, 128}, f32)` for the FP8 dequant scratch with a static `cudaMalloc` device pointer grown lazily (GT-16 style) is a REGRESSION, not a win. Confirmed run on Modal B200: mean 0.803→0.838 ms (+4.4%), p50 0.763→0.787 ms (+3.1%), p95 1.587→1.721 ms (+8.4%), max 1.656→1.797 ms (+8.5%). Only min improved (0.111→0.100 ms, the single-page workload where no grow event fires). Root cause: `max_num_pages` varies non-monotonically across the 128 workloads (range 1–90+), so a single static pointer must `cudaFree`+`cudaMalloc` on every size-increase; `cudaFree` requires a full stream sync, stalling the GPU. torch's caching allocator keeps a per-stream warm pool bucketed by size, so repeat allocations at the same (or smaller) size are effectively free and do not sync. GT-16 applies only to multi-CTA kernels where scratch is directly shared between kernel launches and allocator churn breaks parallelism — it does NOT apply to torch-level scratch in this indexer. Do not attempt this substitution again. Confirmed 2026-04-18 on Modal B200 via full 128-workload run (Modal app ap-GIVP2FstThoEvlcqjMVtOn).

### GT-27: TORCH_CUDA_ARCH_LIST=10.0a Required for tcgen05 (sm_100 rejects all tcgen05 ops)

Any kernel emitting `tcgen05.*` PTX (alloc/dealloc/mma/commit/fence/ld/st/wait) MUST be compiled targeting sm_100a, not generic sm_100. `torch.utils.cpp_extension.load()` selects gencode via the `TORCH_CUDA_ARCH_LIST` environment variable; unset, it auto-detects from the live GPU and on B200 defaults to `sm_100` — which ptxas rejects for every single tcgen05 instruction ("Instruction 'tcgen05.*' not supported on .target 'sm_100'", "Feature '.cta_group::1' not supported", "Feature '.kind::f8f6f4' not supported", "Feature '.32x32b' not supported"). Fix: set `TORCH_CUDA_ARCH_LIST=10.0a` in the Modal image via `.env({"TORCH_CUDA_ARCH_LIST": "10.0a"})`, and defensively re-set it at function entry (`os.environ.setdefault(...)`). Produces `-gencode=arch=compute_100a,code=sm_100a` which ptxas accepts. Confirmed 2026-04-19 on Modal B200 (first Tier 3 attempt COMPILE_ERROR on 128/128, second attempt after env patch PASS 128/128). Applies to all future Tier 3 kernels; no separate per-optimization retry needed — treat as always-on infrastructure.

### GT-46: First-Slab Approximation Hypothesis — EMPIRICALLY REJECTED

D1-D2/V1-V2 problem-space analysis surfaced a new architectural candidate not in prior sweeps: truncate the MMA K-dimension from K=128 (all 4 slabs) to partial K (1-3 slabs), hypothesizing that partial scores might preserve top-K ordering within the rel_err ~1e-2 tolerance (GT-42). Implemented as `tier3_derivation/new_kernel_architecture.cu`. Tested on Modal B200:

| Variant | K used | Mean latency | Correctness |
|---|---|---|---|
| Baseline | 128 (4 slabs) | 0.038 ms | 128/128 PASS |
| 1 slab | 32 | **0.026 ms (−32%)** | **59/128 FAIL** (rel_err up to 7.69) |
| 2 slabs | 64 | 0.026 ms | 59/128 FAIL (same workloads) |
| 3 slabs | 96 | 0.027 ms | 59/128 FAIL (same workloads) |

**Key observation:** the correctness failure set is **stable across K=32, 64, 96**. Exactly the same 59 workloads fail in all three approximation levels. This pattern reveals that the last slab (K=96..128, i.e., dims 96-127) contains information NOT recoverable from any prefix. The errors aren't gradual approximation noise — they're structural.

**Why structural:**
1. The Q/K representations were learned with full K=128 dot products. Trained model semantics don't decompose into early-dim approximations.
2. The ReLU gate creates per-head positive/negative discontinuities. A partial dot product may cross zero in a direction the full product doesn't (false-positive or false-negative ReLU activation per head).
3. rel_err up to 7.69 (=769%) indicates we're not just reordering within top-K — we're selecting genuinely wrong tokens.

**The 32% latency gain is unusable.** The 1-MMA variant showed that removing 3 of 4 MMA slabs delivers a big speedup IF correctness held (it doesn't). Interestingly, the 2-slab and 3-slab variants didn't gain meaningful additional latency over 1-slab — the MMA issue is ~1-2% of kernel time, not the dominant cost (the K HBM load dominates). So even if we could fix correctness with 3 slabs, the latency win would be marginal.

**Implications for all "truncated-K" or "partial-approximation" architectural paths:** the problem's dependency on full K=128 is a hard structural constraint, not a tolerance-loosening opportunity. Any variant that tries to estimate q·K with reduced K dims will fail correctness at this tolerance.

**This closes the architectural exploration opened by D1-D2 paths α, γ, δ, and K** (all relied on some form of partial scoring). The problem genuinely requires all 128 dims.

**The experimental kernel at `tier3_derivation/new_kernel_architecture.cu`** is preserved as a learning artifact. Its final state is the 3-slab variant (59/128 FAIL). Future developers should NOT retry truncated-K approximation unless either (a) the model's Q/K are retrained with dim-locality inductive bias, or (b) the downstream tolerance is relaxed significantly above rel_err ~1e-2.

Kernel reverted to 0.038 ms baseline. No production change. Confirmed 2026-04-19 on Modal B200.

### GT-47: TMEM Double-Buffer (MMA[next] dispatched before H-reduce[curr]) — 69/128 PASS, 59 INCORRECT_NUMERICAL on the SAME 59-workload set as GT-46

Attempted Tier 3 optimization: extend `tcgen05.alloc` from 64 to 128 cols and use [tmem_col, +64) as region A, [tmem_col+64, +128) as region B. Per-page region = `pidx & 1`. Two mbarriers (mbar_A, mbar_B) with independent phase-parity counters. Loop body restructured to dispatch MMA[next_pidx] BEFORE the curr page's mbar wait, so the MMA hardware pipeline of MMA[next] runs concurrent with the H-reduction of MMA[curr]. K-side double buffer (Opt-2) preserved.

**Confirmed run on Modal B200 (app ap-bcupculcr / ap-bdbx63icw):** 69/128 PASS, **59/128 INCORRECT_NUMERICAL** with abs_err 27-170, rel_err 0.5-6275. Latency on passing subset: mean 0.027 ms, p50 0.027, min 0.015, max 0.039 — i.e., **−29% on the workloads that returned correct results**. The optimization mechanism (MMA pipeline overlap) clearly works; the failure is a correctness bug.

**The 59-workload failure pattern is bit-for-bit the SAME set GT-46 saw with truncated-K.** This is a strong empirical signal that these 59 workloads have characteristics (most likely max_num_pages large enough that per-CTA `num_active ≥ 2`, AND a data distribution where mid-page MMA results materially affect the top-K) that exercise a code path the other 69 workloads avoid. The smaller workloads' single-page CTAs only run the prologue path (which keeps the baseline single-buffer behavior), so they never trigger the bug.

**Two race fixes attempted, neither resolved it:**
1. **Race 1 (real, fixed):** Stage B writes `scale_smem_buf[later_K_buf]` where `later_K_buf = (pidx+2) & 1 = pidx & 1 = curr & 1` — same buffer Stage D-1's scale-apply reads. Stage B's scale write was placed BEFORE Stage D-1 in the first attempt, racing across threads (lane<16 of all 4 warps reads, tid<64 of warps 0+1 writes — disjoint thread sets, no implicit ordering). Fix: moved Stage B AFTER Stage D-1's `__syncthreads`. Same-pattern 59 failures after the fix → race 1 was real but not the dominant bug.
2. **Race 2 candidates investigated, none isolated:** elect_one_sync potentially re-electing different threads (commit "tracks all prior MMAs from same thread" — but each commit is in the same elected-thread block as its MMA, so this should be self-consistent), TMEM region B addressing (`(warp_id*32) << 16 | (tmem_col + 64)` — verified per-warp-base format matches GT-12 + tcgen05_ld docs), mbarrier phase parity (manually traced for num_active = 1, 2, 3, 4; all sequences self-consistent), prologue ordering (cp_async_wait_all + __syncthreads before any MMA), SMEM layout (8 B mbar_B added, weights offset shifted to maintain 16-B alignment, total 42048 B < 48 KB). Inspection alone could not isolate the bug.

**What's salvageable / what to try next:**
- The mechanism is real (verified −29% on passing subset). Future attempts should bisect by INCREMENTALLY adding TMEM-double-buffer pieces: (a) alloc 128 cols but use only region A → confirm 128/128, (b) add region B without reordering loop body → confirm 128/128, (c) add 2nd mbarrier without reordering → confirm 128/128, (d) finally add the Stage A reordering. Each step verified in isolation.
- The 59-workload failure correlation with GT-46 is *not* a bug in either change individually; rather, it's the marker that this workload subset exercises any code path that depends on mid-iteration / mid-page state being correct. Future optimizations that touch the page-loop body should expect the same 59 to fail first.
- Buggy attempt preserved at `tier3_derivation/kernel_tmem2_buggy.cu` (post-race-1-fix version) for forensic comparison.

**Closing rule:** Do not attempt TMEM double-buffer + per-iter MMA reorder as one combined change. Bisect via the (a)→(d) sequence above, OR add NCU/printf-based debug instrumentation before the next attempt. Confirmed 2026-04-20 on Modal B200. Kernel reverted to 0.038 ms baseline (verified 128/128 PASS via app ap-3xDZ2hiu1VUEWYR7e086Vw).

**GT-47 follow-up (2026-04-20): TMEM diagnostic probe results — bug surface narrowed dramatically.**

Built `diagnostic_tests/tmem2_probe.cu` + `run_tmem2_probe.py`. Single-CTA, hand-crafted Q/K_A/K_B/K_C giving D_A=I, D_B=2I, D_C=4I as bit-exact expected outputs. 6 probe modes:

| Mode | Tests | Result | max_abs_err |
|---|---|---|---|
| M0 | single MMA → region A | PASS | 0 |
| P1 | single MMA → region B at `tmem_col + 64` | **PASS** | **0** |
| P2 | dispatch MMA[A]→commit, MMA[B]→commit (no wait between), then ld both | **PASS** | **0** |
| P3 | P2 with FORCED single lane (lane 0) for both MMAs | PASS | 0 |
| P4 | same mbar reused twice: commit→wait→commit→wait with phase flip | **PASS** | **0** |
| P5 | production-like 3-MMA flow: MMA[A]→MMA[B]→wait_A→ld_A→MMA[A reused with K_C]→wait_B→ld_B→wait_A→ld_A2 (mbar_A reused, region A overwritten while MMA[B] in flight) | **PASS** | **0** |

**What the probes ELIMINATE as bug suspects:**
- `tcgen05.alloc(128)` returning a usable base column for two regions
- `tcgen05.mma` writing correctly to `tmem_col + 64`
- `tcgen05.ld.32x32b.x32` reading correctly from `tmem_col + 64`
- Two MMAs in flight with different accumulators
- `elect_one_sync` re-electing different threads across calls (forced single-lane gives identical results)
- mbarrier reuse with phase-parity flip after commit→wait→commit→wait
- Region overwrite (writing into a region that was just read by tcgen05.ld) under concurrent in-flight MMA on the other region
- The full mbar_A-reuse + region-A-overwrite sequence that the production loop performs across iters pidx and pidx+2

**What the probes DO NOT yet cover (remaining bug surface for the GT-47 attempt):**
- K_smem buffers loaded via `cp.async` (probe loads K via direct uint4 copies)
- `cp.async.commit` + `cp.async.wait_all` + `__syncthreads` visibility chain feeding into a subsequent MMA's SMEM descriptor read
- Multiple concurrent commit groups for cp.async (Q + K + scale buffers all in flight)
- Multi-CTA effects (probe is single-CTA; production has many CTAs sharing L2/HBM)
- The `scale_smem` synchronous write coexisting with cp.async issue in stage B
- Stage E's `if (has_later)` conditional sync — last 1-2 iters of every CTA skip it

**Probe modes preserved at:** `diagnostic_tests/tmem2_probe.cu` and `diagnostic_tests/run_tmem2_probe.py`. Confirmed 2026-04-20 on Modal B200 (apps ap-zc2h9DFrnaTfVQxfbeKcR3 and ap-QyhlQ405lIqWiqxbH9wehQ).

**Next bisection step (queued):** apply minimal incremental changes to the production kernel — (a) alloc 128 only, (b) +2nd mbar slot wired but unused, (c) +region B actually used in the reorder. First failing step localizes the bug to a specific incremental change. This avoids debating "could it be cp.async?" — the production kernel will tell us directly which step fails.

**GT-47 follow-up #2 (2026-04-20): full bisection on production kernel — bug localized to Stage A reorder but mechanism remains unidentified.**

Performed 6-step bisection on the production kernel, each step validated on Modal B200 against the 128-workload set:

| Step | Change | Result | Mean |
|---|---|---|---|
| (a) | `tcgen05_alloc(128)` instead of (64); rest baseline | **128/128 PASS** | 0.038 ms |
| (b) | + `MBAR_B_SLOT` added to SMEM layout, initialized via `if (warp 0 lane 1)`; UNUSED in loop | **128/128 PASS** | 0.040 ms |
| (c) | + Loop alternates regions/mbars per page (`pidx & 1`) — single MMA in flight, dispatch + commit + wait + ld in same iter | **128/128 PASS** | 0.039 ms |
| (d) | + Stage A reorder restored (full TMEM2 buggy: dispatch MMA[next] before mbar wait of MMA[curr]) | **59/128 FAIL** | 0.027 ms (passing subset) |
| (e) | (d) but with cp.async in Stage B replaced by SYNCHRONOUS uint4 stores | **59/128 FAIL** | 0.027 ms |
| (f) | (e) but with `__launch_bounds__(128, 1)` instead of (128, 2) | **59/128 FAIL** | 0.027 ms |

**Bisection conclusions:**

1. The Stage A reorder is the *only* incremental change that triggers the 59-fail pattern. (a)→(c) all pass; only adding the reorder breaks correctness.
2. **cp.async is NOT the cause.** Bisect (e) eliminated all in-loop cp.async (kept only prologue cp.async); failure is identical.
3. **Per-SM CTA count is NOT the cause.** Bisect (f) restricted to 1 CTA/SM; failure is identical. Bug is intra-CTA.
4. **TMEM mechanics (alloc 128, region B addressing, mbar reuse, region overwrite, two-MMAs-in-flight, elect_one_sync re-election) are NOT the cause.** Probe modes M0/P1/P2/P3/P4/P5 all PASS bit-exactly in isolation.

**What this leaves as the actual cause (untestable in current bisection style):** the Stage A reorder produces a code path where MMA[next] is in flight on the tensor core *concurrent with* the H-reduction loop body (which reads `scores_smem` heavily across all warps) AND the synchronous store path in Stage B (writing `K_smem[curr]` and `scale_smem[curr]`). One of:

  - **scores_smem accesses by H-reduce competing for SMEM bandwidth with MMA[next]'s K_smem reads** (different addresses, same SMEM hardware) — but corruption from bandwidth contention would be very unusual.
  - **Some MMA pipeline state corruption when the elected thread continues into H-reduce while its MMA dispatch is still propagating to the tensor core scoreboard** — would be undocumented hardware behavior.
  - **Specific SMEM bank conflict pattern** that only manifests with the reorder + multi-page workloads + region overwrite combined.

Removing H-reduce, scale_smem write, or K_smem write from the loop body to test these hypotheses would independently break correctness, so further bisection would require **NCU profiling** (smem_bank_conflicts, sm__cycles_active in tensor pipe vs ALU pipe) or **printf-based debug instrumentation** in the kernel itself — neither of which fits the "edit kernel.cu, modal run, compare" cycle the project is built around.

**Final verdict for GT-47:** mechanism is real (passing-subset latency is consistently 0.027 ms vs baseline 0.038 ms = −29%), but the failure mode is below the granularity our bench-only tooling can isolate. The optimization is BLOCKED until either:
- NCU/CUDA-GDB is added to the workflow, OR
- A hardware/PTX-level workaround is found (e.g., explicit `tcgen05.fence::before_thread_sync` between MMA dispatch and subsequent SMEM operations — untested), OR
- The Stage A reorder is restructured to not interleave H-reduce with in-flight MMA[next] (loses the optimization's core mechanism).

**Confirmed terminal:** kernel reverted to baseline (0.038 ms, 128/128) for the third time. GT-47 closed as "diagnosed to bisection floor; no further progress without new tooling." Confirmed 2026-04-20 on Modal B200 via apps ap-AsaVgDvOSIABLoE0ZbEKUA, ap-IgHtgC3Y0EXLkYjQFCdADU, ap-0wg6nU6G2h5hnhuKQSy6bT, plus three more bisect runs whose ids are in `/tmp/bisect_*.log` headers.

**Diagnostic artifacts preserved:**
- `tier3_derivation/kernel_tmem2_buggy.cu` — failing TMEM2 reorder
- `diagnostic_tests/tmem2_probe.cu` + `run_tmem2_probe.py` — 6-mode probe (all pass, can be reused for future tcgen05 work)
- `/tmp/bisect_{a,b,c,d,e,f}.log` — Modal logs for each bisection step (kept for forensic reference)

**GT-47 follow-up #3 (2026-04-20): P6-P10 production-replica probes — root cause identified as undocumented multi-CTA hardware contention; unfixable without NCU/hardware-team support.**

Built a separate production-replica probe at `diagnostic_tests/tmem2_p6_production_replica.cu` + `run_tmem2_p6.py` that mirrors the production buggy loop body (Stage A reorder + H-reduce + scale apply + scores_smem + GMEM write) on synthetic Q/K data with bit-exact expected outputs (D_p[m,n] = δ_{m,n}·value_p, weights=1, scale=1 → final_scores[p*64..(p+1)*64] = value_p). Tested 5 progressive variants:

| Probe | Variant | Result |
|---|---|---|
| **P6** | Single CTA (B=1), full production loop body | **PASS** bit-exact |
| **P7** | Multi-CTA (B=32) launch (1, B), shared inputs | **FAIL** 11/32 wrong CTAs, max_err 1.0 |
| **P7b** | Multi-CTA (B=128) | **FAIL** 27/128 wrong CTAs, max_err 3.0 (matches K_C value) |
| **P8** | P7 + `tcgen05.fence::before_thread_sync` after `wait::ld` | **FAIL** 19/32 (slightly worse) |
| **P9** | P8 + `tcgen05.fence::after_thread_sync` before next iter's MMA dispatch | **FAIL but partial** 13/32, first wrong CTA's wrong cells dropped 16→6 |
| **P10** | P9 + 4 TMEM regions (no reuse across pages, alloc 256 cols, 4 mbars) | **FAIL** 14/32 |

**Pinpoint pattern from P7 detail dump (consistent across all failing variants):**
- Always **page 0** (the prologue's MMA[0])
- Exactly **16 wrong cells in page 0**: positions `[34, 35, 38, 39, 42, 43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63]`
- All in **HI half of cols** (read by `taddr_hi` = `(warp_id*32) << 16 | (tmem_col + 32)`)
- Row == col (diagonals); rows ≥ 32 → only **warps 2 & 3** affected
- Within warps 2/3, only lanes where `(lane_offset & 2)` is set
- Wrong value at B=128 = **4.0 = page 2's K_C value** (page 2 reuses TMEM region A in 2-region design)
- Wrong value at B=32 = **0.0** (timing-dependent: MMA[2] hasn't yet written, ld picks up zeros)

**Critical finding from P10:** even with **4 dedicated TMEM regions and zero region reuse**, the bug persists with the same 16-position pattern and the same "wrong value matches page 2's K" — except now there's no shared region between page 0 and page 2 to physically conflict over. This rules out:
- Region reuse race (P10 has none)
- TMEM addressing/aliasing
- mbarrier reuse race (P10 has 4 distinct mbars)
- cp.async-MMA race (already ruled out by bisect (e))
- Per-SM CTA count (already ruled out by bisect (f))
- elect_one_sync re-election (ruled out by P3)

**Remaining root cause hypothesis (untestable with current tooling):** an **undocumented per-SM hardware contention** in the tcgen05 tensor core scheduler when many CTAs each have **multiple in-flight MMAs from a single thread** (the elected thread of warp 0 dispatches MMA[next] while MMA[curr] is still pending). The contention manifests as MMA[0]'s output (warps 2/3, HI half cols) being partially polluted by data from MMA[2] (which uses K[2] = K_C with value 4.0).

The pattern's stability across variants — same 16 positions, same wrong value tracking page 2's K — suggests the tensor core is delivering MMA[0]'s `tcgen05.ld` result with some slots pulling MMA[2]'s data, even when:
- MMA[2] writes to a completely different TMEM region (P10)
- Explicit fence pair surrounds the ld→mma transition (P9)
- Synchronous K loads are used (bisect e)
- Only 1 CTA/SM is allowed (bisect f)

This is below documented PTX-level semantics. Verification would require:
- NCU `sm__cycles_active.tensor_pipe` and `tcgen05_*` counters
- Or dialog with NVIDIA's tcgen05 hardware team about per-SM tensor-pipe scoreboard limits

**Final GT-47 verdict: BLOCKED — root cause identified as multi-CTA tensor core scheduler contention; production-side mitigation requires either (a) NCU profiling we don't have, (b) limiting to 1 in-flight MMA per CTA (which is bisect (c) = baseline = no perf gain), or (c) a hardware/PTX-level workaround discovered later.** The diagnostic infrastructure built (P6-P10 probes + run_tmem2_p6.py) is reusable for any future investigation. Confirmed 2026-04-20 on Modal B200 across apps ap-nKoI2zrf9RuFyBZ80ohKQh, ap-8vNGfgHxmkhafGs0TXmmPW, ap-5XrvQDCnjUwqrGxwF5aU4Z, ap-ZTzmlKDnKny8GIur64REv5, plus P9/P10 runs whose ids are in `/tmp/p{8,9,10}.log`.

**Additional artifacts:**
- `diagnostic_tests/tmem2_p6_production_replica.cu` — production-replica kernel with all probe modes
- `diagnostic_tests/run_tmem2_p6.py` — Modal runner with detailed wrong-position dump
- `/tmp/p{6,7,7_detail,8,9,10}.log` — per-probe Modal logs (forensic reference)

### GT-48: tcgen05.commit mbar fires before MMA fully drains — undocumented hardware behavior, applies to ALL future tcgen05 work

**This GT supersedes GT-47's "blocked" verdict with a concrete root cause and fix.**

A new probe at `diagnostic_tests/tmem2_p11_kmem_overwrite_race.cu` (P11) isolated the GT-47 root cause to a single sentence: **`tcgen05.commit`'s mbarrier arrival fires before the tensor pipe has fully drained — both TMEM writes AND source-SMEM reads can still be in flight when `mbarrier.try_wait` returns.** The PTX ISA does not document this gap.

**Direct empirical proof from P11 (2026-04-20, Modal app ap-lX7nudaFtLlEzoAZVtPw84):**

| Test | Setup | Result | Diagnosis |
|---|---|---|---|
| P11a | Single CTA, no overwrite, no `__syncthreads` after `wait_ld` | **FAIL: 16 zeros at GT-47 positions** | TMEM writes incomplete when mbar fires — 16 specific cells (warps 2/3, HI half cols, lane bit-1 set) are not yet written |
| P11b | Single CTA, K_smem overwrite + `__syncthreads` between mbar wait and ld | **PASS bit-exact** | The `__syncthreads` provides enough drain time in single-CTA |
| P11c | B=32, K_smem overwrite + `__syncthreads` | **FAIL: 8 cells = K_C value (4.0)** at positions [35, 39, 43, 47, 51, 55, 59, 63] | Multi-CTA contention keeps MMA's source-SMEM reads in flight past `__syncthreads`; the K_smem overwrite physically replaces K_A bytes with K_C; MMA's still-pending reads pull K_C; output = `Q · K_C^T` = 4.0 on diagonal |
| P11d | B=128, same as P11c | Same 8-position K_C pattern, all 128 CTAs FAIL | Confirms multi-CTA at scale |

**Position pattern is hardware-deterministic and reflects TMEM physical structure:**
- Affected lanes: `(lane_offset & 2)` set within warps 2 & 3 (rows 32-63)
- HI half of MMA output (cols 32-63 read by `taddr_hi`)
- The "(lane & 3) == 3" subset persists even with `__syncthreads` partial drain — these are the LAST lanes in each 4-lane physical group to drain

**Production fix (verified 2026-04-20 on Modal B200, app `brw1bbs9h`):**

Add `__syncthreads()` immediately after `tcgen05_wait_ld()` in stage C of the TMEM2 reorder kernel. Result: **128/128 PASS, mean 0.039 ms** (matches baseline within noise).

```cpp
tcgen05_ld_32x32b_x32(taddr_lo, rlo);
tcgen05_ld_32x32b_x32(taddr_hi, rhi);
tcgen05_wait_ld();
__syncthreads();   // GT-48: tensor pipe drain — required after wait_ld when
                   // any subsequent op may overwrite the MMA's source SMEM
                   // (or the MMA's own TMEM region with another in-flight MMA).
```

**Honest perf assessment of TMEM2 + GT-48 fix:**
- Baseline (no TMEM2): 0.038 ms mean / 128/128 PASS
- TMEM2 buggy: 0.027 ms passing-subset mean (BIASED — only smaller workloads passed) / 59 fail
- TMEM2 + GT-48 fix: 0.039 ms mean / 128/128 PASS — **no perf gain over baseline**

The buggy version's "0.027 ms" overstated the optimization's potential because the 69 passing workloads were systematically smaller. The required `__syncthreads` to maintain correctness eats the H-reduce/MMA-pipe overlap that TMEM2 was supposed to deliver. The TMEM2 architectural mechanism is real but the practical perf gain is zero on this workload distribution. **Production kernel reverted to baseline (0.038 ms, 128/128).**

**Generalizable rule (the GT proper):** any tcgen05 kernel that uses the `tcgen05.commit → mbarrier::arrive::one → mbarrier.try_wait` pattern AND subsequently overwrites the MMA's source SMEM (K_smem reuse) OR dispatches another MMA to the same TMEM region MUST add `__syncthreads()` after `tcgen05.wait::ld` (or equivalent tensor-pipe-drain barrier) before any such subsequent op. The PTX docs say `tcgen05.commit` "tracks completion" — empirically, completion is signaled before all hardware effects propagate. The `__syncthreads` is the cheapest available drain primitive; alternatives (dummy second commit/wait cycle, `__threadfence_block`, delay loop) are also possible.

**Why this matters for future Tier 3 work:**
- ANY future kernel reusing K_smem buffers across MMAs is at risk of GT-48 silently corrupting MMA outputs — the corruption pattern is deterministic at the same 16 positions and may go unnoticed if not specifically tested at scale
- The single-CTA test (M0 in tmem2_probe.cu) PASSES even without the drain barrier because timing is tight; the bug only manifests at multi-CTA scale
- Always test new tcgen05 work with `B ≥ 32` parallel CTAs to catch this class of bug

**Diagnostic infrastructure (reusable):**
- `diagnostic_tests/tmem2_p11_kmem_overwrite_race.cu` — minimal repro
- `diagnostic_tests/run_tmem2_p11.py` — Modal runner with K_C contamination detection
- The 4-mode test (control / single-CTA-with-overwrite / multi-CTA / multi-CTA-large) is the canonical detector for this hardware behavior

Confirmed 2026-04-20 on Modal B200 across apps ap-lX7nudaFtLlEzoAZVtPw84 (P11 isolation) and `brw1bbs9h` (production fix verification 128/128 PASS).

**GT-48 follow-up: drain barrier is positionally AND functionally rigid (A/B/C investigation 2026-04-20).** Three optimization candidates were tested against tcgen05_kernel.cu (the saved 2-region+drain variant at 0.039 ms) to try to recover the H-reduce / MMA-pipe overlap that the drain consumes:

| Candidate | Mechanism | Result | Lesson |
|---|---|---|---|
| A | Move drain past H-reduce (right before stage B's K_smem write) | **FAIL 59/128** with the GT-47 pattern | Drain MUST be IMMEDIATELY after `wait_ld`. The barrier itself triggers tensor-pipe quiescence that must precede any subsequent op (scale apply, H-reduce, etc.), not just K_smem write |
| B | Replace `__syncthreads` with `__threadfence_block` (memory ordering only) | **FAIL 59/128** with the GT-47 pattern | Drain requires real `bar.sync` (thread synchronization). Memory ordering alone (no thread sync) does NOT trigger the tensor-pipe quiescence |
| C | 4 TMEM regions (alloc 256, no region reuse) + drain at working position | **PASS 128/128 @ 0.040 ms** | Correctness alternative; no perf gain. GT-48 drain remains the bottleneck regardless of region count |

**Refined GT-48 statement:** the drain barrier is doubly rigid — (1) **positionally**: must be IMMEDIATELY after `tcgen05_wait_ld()`, not later in the loop body, AND (2) **functionally**: must be `bar.sync` (thread synchronization), not memory ordering. This narrows fix options significantly. The GT-48 fix achieves correctness; recovering the originally-intended TMEM2-reorder perf gain (~5-10%) requires a path that doesn't go through the drain at all — likely candidate D (restructure so a separate warp does K_smem overwrite, allowing per-warp drain via named bar with subset) or candidate F (NCU profiling to find the specific pipe stall and a non-obvious workaround). Both untested.

**GT-48 follow-up #2: Cand D + E tested 2026-04-20 — drain mechanism is TRIPLY rigid.** Two more drain-mechanism candidates were tested:

| Cand | Mechanism | Result | Lesson |
|---|---|---|---|
| D | Replace `bar.sync 0` (= __syncthreads) with named `bar.sync 1, 128` | **PASS 128/128 @ 0.040 ms** | Drain triggers via ANY `bar.sync` instruction, not specifically barrier ID 0. No perf differentiation between barrier IDs. |
| E | Prefix __syncthreads with explicit `tcgen05.fence::before_thread_sync` (PTX canonical pattern) | **PASS 128/128 @ 0.040 ms** | The fence is redundant when paired with bar.sync — bar.sync alone provides equivalent ordering. No perf gain from the explicit fence. |

**Final GT-48 verdict (after A-E investigation):** the drain barrier is TRIPLY rigid:
1. **Position**: must be IMMEDIATELY after `tcgen05_wait_ld()` (Cand A failed)
2. **Function**: must be `bar.sync` (thread sync), not memory ordering (Cand B failed)
3. **Cost**: any `bar.sync` variant has equivalent cost — barrier ID, fence prefix, or alternate region count don't change perf (Cand C, D, E all 0.040 ms)

The TMEM2-reorder optimization is **fundamentally perf-blocked at the GT-48 drain** — no available drain-mechanism variant we've tested is light enough to recover the H-reduce/MMA-pipe overlap. The optimization mechanism is real (verified in single-CTA P6) but the multi-CTA correctness requirement (drain via `bar.sync`) costs as much as the optimization saves. The kernel at `tier3_derivation/tcgen05_kernel.cu` is finalized at 0.039 ms / 128/128 with the canonical drain placement, preserved for future structural-rework candidates (F: split-warp K_smem write with per-warp drain; G: NCU profiling; H: different MMA shape). All five A-E candidate variants documented in the kernel header table.

**GT-48 follow-up #3: Cand F1 (split-warp drain) tested 2026-04-20 — drain is QUADRUPLY rigid (scope must be full-CTA).**

| Cand | Mechanism | Result | Lesson |
|---|---|---|---|
| **F1** | Subset `bar.sync 1, 64` called only by warps 0-1 (64 threads) instead of full-CTA __syncthreads | **FAIL 59/128 (GT-47 pattern)** | Per-warp-subset drain does NOT trigger SM-wide tensor pipe quiescence. The drain mechanism requires ALL 128 threads of the CTA to participate in the barrier. Closes the split-warp design path: cannot decouple warps for tensor pipe purposes. |

**Final GT-48 verdict (post A-F1):** the drain is **QUADRUPLY rigid:**
1. **Position**: immediately after `wait_ld` (Cand A failed)
2. **Function**: `bar.sync` (Cand B failed for `__threadfence_block`)
3. **Cost**: invariant across `bar.sync` variants (Cand D, E confirmed)
4. **Scope**: full-CTA — all 128 threads must participate (Cand F1 failed for 64-thread subset)

This closes the per-warp-subset drain path. Workarounds requiring "warps 2-3 do useful work in parallel during drain" (Cand F2 split-work design) are blocked since warps 2-3 must also participate in the drain barrier — they cannot be doing other work during it.

**Remaining untested paths (require external tooling or fundamental rework):**
- **Cand G (NCU profiling)**: query tensor-pipe counters to find the specific pipe stall and a non-obvious workaround. Modal supports `ncu` CLI.
- **Cand H (different MMA shape)**: M=128 might change tensor pipe scheduling pattern. Risky, requires rederivation.
- **Cand I (cluster + DSM)**: sm_100a clusters may share TMEM across CTAs, possibly avoiding per-CTA drain. GT-44 saw a regression with a different cluster design; might be worth retrying.
- **Cand J**: revert the Stage A reorder entirely (it's essentially equivalent to baseline if the optimization can't be recovered).

The TMEM2-reorder kernel at `tier3_derivation/tcgen05_kernel.cu` is finalized at 0.039 ms / 128/128. All seven candidates A through F1 documented in the kernel header.

### GT-49: Cluster G=2 top-K with DSM merge — REGRESSION (+5% mean) regardless of merge algorithm

After NCU profiling showed top-K is **35% of kernel time** at the largest workload (B=30, max_num_pages=91), tested two cluster-based G=2 top-K variants on Modal B200:

| Variant | Phase 2 mechanism | Result |
|---|---|---|
| Re-sort merge | CUB BlockRadixSort on combined 2K=4K candidates | 128/128 PASS @ 0.040 ms (+5% mean, +12% max) |
| **Parallel merge-path** | O(log K) binary search per output position; 8 outputs/thread | **128/128 PASS @ 0.040 ms (same regression)** |

Both variants use `__cluster_dims__(2,1,1)` with 256 threads/CTA. Phase 1: each CTA sorts its half of N_max via CUB. cluster.sync. Phase 2: cta_rank=0 reads other CTA's partial via `cluster.map_shared_rank`, computes top-K, emits with index translation. Dispatched only for N_max ≥ 4096.

**Why both fail despite parallel merge-path being O(K) (=~0.5 µs Phase 2 wall) vs re-sort's O(K log K) (=~21.8 µs):**
- Cluster overhead dominates: cluster.sync × 2 likely 5-10 µs; cluster placement constraints force resource sharing
- Per-CTA sort time may not scale linearly with N (CUB radix has fixed per-pass overhead)
- 35% top-K cost from NCU was on the largest single workload with NCU's own overhead inflating; benchmark mean (0.038 ms) has different cost mix where top-K is a smaller absolute fraction
- Top-K work is too small relative to cluster overhead at our K and B values

**Cluster mechanics confirmed working** (128/128 PASS = correctness preserved across both merge approaches, plus the GT-44 cluster work). cluster.sync, `cluster.map_shared_rank` for DSM addressing, `__cluster_dims__` launch — all sound on sm_100a.

**Implication for Cand 3 (full Streaming Gate via cluster+DSM):** the cluster overhead seen here would also apply to Cand 3, possibly making its expected savings less attractive than analysis suggested. The HBM benefit of eliminating final_scores round trip (~720 KB = ~90 ns at 8 TB/s) is negligible per NCU; remaining savings would only come from kernel launch overhead (~1-3 µs), which cluster overhead would consume. Cand 3 expected savings revised from "25-45%" down to "uncertain, possibly negative."

**Code preserved at** `flashinfer-bench-starter-kit/solution/cuda/kernel.cu` (variant `block_topk_translate_cluster_kernel`, lines ~660-805) for future reference. Dispatch reverted to baseline path. Kernel back at 0.038 ms / 128/128. Confirmed 2026-04-20 on Modal B200 across apps for both re-sort and parallel-merge-path variants.

### GT-50: Persistent-CTA scoring variant for slow workloads — NULL on mean, REGRESSION on tail

Tested workload-specialized persistent kernel `fp8_mma_final_scores_persistent_kernel` dispatched only when `total_work = B * ceil(mnp/4) > 296` (= 2× B200 SMs). Design: launch fixed 296 CTAs (matching `__launch_bounds__(128, 2)` × 148 SMs), block-partition `total_work` items across CTAs to maximize Q-load reuse on consecutive same-batch work items, amortize TMEM alloc + mbar init over CTA-life rather than per-CTA-launch.

Confirmed run on Modal B200 (app ap-eYB4YqKxgNo67hCby5v1pj): **128/128 PASS, mean 0.039 ms (unchanged)**, but the slow workloads (B≥27, mnp≥80) all REGRESSED ~5-10%:

| Workload | Baseline | Persistent | Δ |
|---|---|---|---|
| bb0f8277 (B=30, mnp=89) | 0.078 | 0.082 | +5% |
| a876010b (B=29, mnp=89) | 0.077 | 0.082 | +6% |
| 8635db8f (B=27, mnp=91) | 0.075 | 0.083 | +11% |
| f362edf4 (B=30, mnp=83) | 0.075 | 0.082 | +9% |
| f7f61b05 (B=14, mnp=91) | 0.058 | 0.066 | +14% |

**Why persistent didn't help on the workload pattern it was designed for:**

The mechanism analysis assumed persistent saves `setup_cost × (waves - 1)` by amortizing TMEM alloc + Q load + weights load over multiple work items per CTA. But empirically the savings were eaten — and then some — by:

1. **Reduced SM occupancy.** Block partition with `items_per_cta = ceil(total_work/296)` means only `ceil(total_work/items_per_cta)` CTAs do real work. For total_work=690, items_per_cta=3 → 230 CTAs busy, 66 idle. 230/148 SMs ≈ 1.55 CTAs/SM avg vs baseline's 2 CTAs/SM in waves 1-2. Lower per-SM concurrency → less latency hiding.
2. **Per-work-item sync overhead.** Each work item needs `__syncthreads` for bt_cache visibility (warp 0 t0 writes, all 128 threads read). Across 3 iterations per CTA, that's ~6 extra syncs not present in baseline (where setup syncs are paid once).
3. **Q-reuse savings tiny.** Q is 8 KB, cp.async load is ~500 cycles. Skipping Q reload on consecutive same-batch items saves only ~500 cycles per skip, dwarfed by the ~25000 cycles per work item.
4. **Wave count wasn't actually reduced.** Original 690 CTAs / 296 slots = 2.3 waves = 3 sequential CTA-times. Persistent: 296 CTAs each doing 3 iterations = 3 sequential per-item-times. Same wall time, no wave reduction.

**Generalizable rule:** persistent CTAs only beat the standard launch when **per-CTA setup cost is a meaningful fraction of per-CTA work cost AND wave count actually drops**. For this kernel, setup is ~3% of per-CTA cost (TMEM alloc ≈ 100 cycles, mbar init ≈ 20 cycles, Q load ≈ 500 cycles, weights load ≈ 50 cycles vs 25000 cycles per work item). Persistent's amortization saves at best 2-3% per CTA, which gets canceled by the 5-10% per-work-item overhead introduced by the per-iteration syncs and reduced occupancy.

**When persistent WOULD work for this codebase pattern:**
- Kernels where setup is >20% of per-CTA cost (e.g., kernels with expensive per-CTA prefix-scan or codec init)
- Kernels with variable per-work-item cost where dynamic atomic-based work distribution beats static partition
- Kernels where wave count reduction is achievable (e.g., reducing per-item SMEM enables more CTAs/SM)

None apply here. Kernel reverted to baseline (0.039 ms / 128/128 / 100% wins vs flashinfer / 3.75x mean speedup). Backup of pre-attempt state saved at `tier3_derivation/kernel_pre_persistent_baseline.cu`.

Confirmed 2026-04-21 on Modal B200.

### GT-45: Final Non-Precision Candidates C + E — C slight regression, E architecturally blocked

Per the GT-44 state analysis, C (multi-stage cp.async depth 3) and E (warp specialization) were flagged as the two remaining viable candidates. Both evaluated:

**C: Triple-buffer cp.async pipeline** — implemented, tested, slight regression (+2.6% mean). 128/128 PASS after fixing a subtle cp.async.wait_group(1) edge case: when only 1 group is pending, wait_group(1) returns immediately without waiting. Required prologue branch: `if (num_active >= 2) wait_group(1)` else `wait_all`. SMEM grew 42032→50480 B (+20%), required `cudaFuncSetAttribute` for >48 KB dynamic SMEM. Latency regressed because PAGES_PER_CTA=4 caps the overlap benefit (only 1 extra overlap stage beyond depth-2) and the SMEM growth cuts L2/occupancy marginally. Reverted.

**E: Warp-specialized scoring** — NOT implemented; architectural analysis showed it cannot yield a real win:
- GT-12: tcgen05.ld on kind::f8f6f4 distributes M-rows across ALL 4 warps (16/warp), not just warp 0.
- GT-17: tcgen05.ld.sync.aligned requires all 32 lanes of ISSUING warp to participate.
- Combined: warps 0-3 are structurally locked into the tcgen05.ld pipeline. Cannot be specialized into "consumer-only" roles.
- TMA for K ruled out by GT-9 → producer warps have no special TMA path (just cp.async like everyone else).
- cp.async is HBM-bound, not thread-bound → adding "producer" warps doesn't accelerate K loads.
- 256-thread variant would add register pressure without more parallel work.

GT-43's estimate of "1-3% yield, 150+ LOC" was optimistic; the real yield is likely 0%. Skipped to avoid 150 LOC of structured code that cannot beat the architectural ceilings.

**Cumulative from all sessions:** kernel at **0.038 ms** (GT-43 baseline). No change. Baseline preserved at `tier3_derivation/kernel_current_baseline.cu`.

**Final state of optimization surface:** per our own GT record, we have now either tested, skipped with evidence, or documented architectural barriers for every enumerated candidate. The only remaining optimization paths require external changes:
- Harness-level (batched evaluation, different timing methodology)
- Problem-spec-level (smaller K, looser tolerance than rel_err~1e-2)
- New hardware (sm_110+ if/when new tcgen05 variants appear)

**Cumulative from GT-BASELINE:** 0.803 → **0.038 ms (−95.3%)**. Confirmed terminal for this kernel given current constraints. Confirmed 2026-04-19 on Modal B200.

### GT-44: sm_100a Cluster + DSM Programming Model — WORKS (correctness) but REGRESSES latency (expected)

Implemented a clustered rewrite as a learning exercise for sm_100a cluster + distributed shared memory (DSM). File: `tier3_derivation/clustered_kernel.cu`. Confirmed run on Modal B200 (app ap-EuNa7Zs7sJ1CQ3EBbyI4Kv): 128/128 PASS with abs_err=0.00 rel_err=0.00, but mean latency 0.038 → **0.095 ms (+150% regression)**. This validates the programming model AND empirically confirms GT-34's architectural analysis — cluster-based fusion at this problem's parameters (K=2048, max_num_pages up to 90) cannot beat the non-clustered design.

**Design:**
- Grid: `(B * CLUSTER_SIZE, 1, 1)` with `__cluster_dims__(2, 1, 1)` — 2 paired CTAs per batch
- Each CTA handles `max_num_pages/2` pages, writes local run_scores to SMEM
- After scoring, `cluster.sync()` synchronizes both CTAs
- CTA 0 (within cluster) reads CTA 1's run_scores via DSM using `mapa.shared::cluster` + `ld.shared::cluster.f32`
- CTA 0 runs CUB BlockRadixSort on combined 5760 scores, writes topk_indices directly
- Final `cluster.sync()` to keep CTA 1's SMEM alive until CTA 0 finishes reading
- Fuses scoring + top-K into one kernel, eliminates final_scores[B, N_max] GMEM round trip

**What WORKED (learning outcomes — these are the valuable GTs from this exercise):**

1. **Cluster launch via `__cluster_dims__` attribute** with standard `<<<grid>>>` syntax works on sm_100a. Grid dim must be divisible by cluster dim — our `B * CLUSTER_SIZE` satisfies this trivially.
2. **DSM addressing via `mapa.shared::cluster.u32`** works: takes (local_smem_addr_u32, target_block_rank_u32) and returns a cluster-shared u32 address valid for other CTAs in the cluster.
3. **DSM loads via `ld.shared::cluster.f32`** work with the cluster-shared addresses. Cannot be dereferenced as regular C++ pointers; must use explicit PTX.
4. **`cluster.sync()`** from `cooperative_groups` works as a cluster-wide barrier. Needed twice: once to ensure all CTAs finished writing before the merge, and once at kernel end to prevent CTA exit while another CTA is still doing DSM reads.
5. **tcgen05 lifecycle (alloc/mma/commit/wait/dealloc) works INSIDE clustered CTAs.** No special treatment needed — each CTA gets its own TMEM allocation normally. The `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64` works with cluster-visible mbarriers (though we kept mbarriers local to each CTA).
6. **SMEM opt-in above 48 KB** via `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 54816)` works for clustered kernels same as non-clustered.
7. **CTA exit ordering matters for DSM.** If CTA 1 exits before CTA 0 finishes DSM reads, CTA 1's SMEM can be deallocated mid-read. A trailing `cluster.sync()` prevents this.

**What did NOT work (architectural lesson):**

The design CORRECTLY eliminates the final_scores GMEM round trip and the separate top-K launch, but at the cost of **collapsing 23 CTAs/batch down to 2 CTAs/cluster-per-batch** → SM utilization drops ~11×. For B=32 workloads, baseline runs ~736 total CTAs; clustered runs only 64 CTAs on 148-SM B200. SMs starve for work; the per-CTA work balloons from ~4 pages to ~45 pages (serialized within CTA).

**Observed breakdown:**
- Small workloads (max_num_pages ≤ 8): regressed ~50-80% (single-cluster execution serializes what was originally parallel across CTAs)
- Large workloads (max_num_pages ≥ 60): regressed ~100-200% (per-CTA scoring loop is now 30+ iterations serialized)

**Why the DSM savings don't compensate:**
- Eliminated GMEM: ~1 MB at B=32 / 8 TB/s = 0.125 µs savings per invocation
- Eliminated launch: ~1-2 µs per invocation
- Total savings: ~1-2 µs
- Cost from lost parallelism: ~50-60 µs (dominant)
- Net: ~+50 µs regression, matches observed

**The generalizable insight:** CTA cluster fusion is only beneficial when the cluster ADDS parallelism (e.g., wider tensor core tiles via cta_group::2 MMA) rather than SUBSTITUTING for existing parallelism. For memory-light, per-batch-independent problems like DSA top-K, cross-CTA state sharing doesn't pay off because there's no shared state that's expensive to recompute. For attention kernels where Q×K intermediate matrices are LARGE and expensive, clusters+DSM is a fit. For this kernel, the scoring output per CTA is tiny (256 scores) compared to the parallelism it would sacrifice.

**Cluster + DSM only wins when:**
- The shared state is larger than the parallelism it sacrifices (e.g., Q×K attention matrix in prefill)
- OR the operation naturally requires cross-CTA cooperation (e.g., a single large matmul tiled across CTAs)
- OR the problem has low intrinsic parallelism such that collapsing CTAs doesn't hurt occupancy

None of these apply to DSA TopK at these parameters.

**Reference kernel preserved** at `tier3_derivation/clustered_kernel.cu` for future sm_100a cluster work. The PTX patterns (mapa.shared::cluster, ld.shared::cluster, cluster.sync pattern, __cluster_dims__ with <<<>>> syntax) are transferable to any future clustered kernel development on this codebase. Kernel reverted to 0.038 ms baseline; no latency change from learning exercise.

Confirmed 2026-04-19 on Modal B200.

### GT-43: 12-Candidate Non-Precision Sweep — Cumulative −2.6% to 0.038 ms

After GT-42 ruled out precision-reducing options, tested 12 non-precision-reducing candidates (A-L).

| ID | Candidate | Result | Mean | Notes |
|---|---|---|---|---|
| A | Hoist Q SMEM descriptors outside page loop | REGRESSION | 0.040 | +2.6%. 4 uint64 descriptors at kernel scope added reg pressure across all 128 threads even though only warp 0 thread 0 used them. Reverted. |
| B | Adaptive PAGES_PER_CTA (2 for small, 4 for large) | NEUTRAL | 0.039 | Dispatch overhead ate small-workload win; few workloads hit the threshold. Reverted. |
| C | Multi-stage cp.async depth 3 | SKIPPED | — | +16 KB SMEM for 3rd K buffer; only 4 pages/CTA limits overlap ceiling to ~2 pages. Expected win <2% disproportionate to ~200 LOC rewrite. |
| **D** | **Extra-small top-K variant (128×4=512 cap for N_max ≤ 512)** | **TAIL WIN** | 0.039 | **min −21% (0.019→0.015)**. Mean flat (smallest workloads don't move the aggregate). Kept for clean tail-latency improvement with no downside. |
| E | Warp-specialized scoring kernel | SKIPPED | — | 150+ LOC rewrite with high risk to tcgen05 lifecycle correctness. |
| F | cudaGraph capture/replay | SKIPPED | — | Tensor pointers change per invocation; graph would need re-instantiation per call, negating savings. |
| G | Cluster DSM Q sharing | SKIPPED | — | ~200 LOC + cluster/DSM infrastructure; Q HBM savings are already <0.025 µs per call. |
| H | cp.async.bulk (TMA) for Q | SKIPPED | — | Host tensormap + CUtensorMap wiring for an 8 KB one-time load; setup overhead > expected gain. |
| I | Pre-compute K buffer descriptors | SKIPPED | — | Same reg-pressure pattern as A (regressed); 8 descriptors would regress more. |
| **J** | **Remove end-of-iter __syncthreads (keep only post-scale-apply)** | **WIN** | **0.038 (−2.6%)** | mbarrier_wait + warp 0's MMA-issue coupling creates a natural barrier: other warps can't pass mbarrier_wait of iter N+1 until warp 0 issues+commits MMA, and warp 0's elected thread also participates in H-reduction of iter N, so it can't issue iter N+1's MMA until iter N's H-reduction is done. No race on scores_smem. 128/128 PASS, abs_err=0.00. Kept. |
| K | (128, 3) launch_bounds + reduced ILP for reg budget | REGRESSION | 0.039 | +2.6%. (128, 3) hint didn't actually deliver occupancy gain, and reduced 4-way ILP (vs 8-way) cost a bit. Reverted. |
| L | Remove TORCH_CHECK + AT_CUDA_CHECK host-side | NEUTRAL | 0.038 | No measurable gain; safety check worth keeping. Reverted. |

**Cumulative this session:** 0.039 → **0.038 ms (−2.6%)** vs GT-41 baseline. **Cumulative from GT-BASELINE (0.803 ms): −95.3%.**

**Key learnings:**
- **Kernel-scope variables pressure all threads (A, I).** Even when only one thread uses them, CUDA allocates regs per thread. Use warp-local or elected-thread-scope where possible. Future candidates that hoist state outside loops should go inside `if (warp_id == 0 && lane_id == 0)` or be `__shared__`.
- **J (barrier removal) worked via implicit tcgen05 coupling.** The analysis: "mbarrier_wait gates non-issuing warps on the issuing warp's MMA dispatch, and the MMA-issuing thread (tid 0) also participates in H-reduction, so it self-gates." This works specifically because our scoring kernel has warp 0 elected thread doing both MMA issue AND H-reduction. Generalizable rule: when an elected-thread sync primitive cascades through MMA-commit to mbarrier_wait, it CAN substitute for explicit __syncthreads IF the elected thread also participates in the data producer step. Logged for future kernels.
- **D (tiny top-K variant) doesn't move the mean but improves the p-min.** Useful for workloads that care about tail latency but not aggregate throughput. Kept for the clean min improvement.
- **(128, 3) occupancy fantasy failed again (K).** Tried twice now (first in GT-41, again with reduced ILP in GT-43). (128, 2) is the sweet spot for this kernel. Compiler either can't fit 3 CTAs/SM without spills, or 3 CTAs/SM add contention (TMEM allocator, SMEM banks) that negates occupancy gain.
- **Most architectural candidates (C, E, F, G, H) got skipped due to complexity vs expected return.** At 0.038 ms with 50%+ harness overhead, the cost-per-percentage-point is now very high. Any future candidate that requires >100 LOC should have a confident expected win of >5% to justify.

Kernel snapshot at `tier3_derivation/kernel_current_baseline.cu`. **Cumulative across all sessions: 0.803 → 0.038 ms (−95.3%).** Confirmed 2026-04-19 on Modal B200.

### GT-42: M1.6 BF16/FP16 scores_smem — CORRECTNESS FAIL on both, test tolerance discovered

Attempted to halve scores_smem footprint (16640 B → 8320 B) by replacing FP32 with BF16 / FP16. H-reduction accumulator stays FP32 (only the SMEM intermediate is lowered). SMEM layout shifts: SCORES_SMEM at 24576 (8320 B), SCALE0 at 32896, ... total kernel SMEM 42032 → 33712 B (−20%).

**Empirical results on Modal B200:**

| Variant | Mean latency | Correctness | Max rel_err (fails) |
|---|---|---|---|
| FP32 baseline | 0.039 ms | 128/128 PASS, abs_err=0.00, rel_err=0.00 | n/a |
| BF16 | **0.038 ms (−2.6%)** | **112/128 PASS, 16 FAIL** | up to 1.38e-02 |
| FP16 | 0.042 ms (+7.7%) | **126/128 PASS, 2 FAIL** | up to 1.07e-02 |

**Key finding: test tolerance discovered via failing runs.** Previously all wins had produced `abs_err=0.00e+00 rel_err=0.00e+00` (bit-identical). The BF16/FP16 failures reveal the harness tolerance threshold sits at approximately **rel_err ~1e-2**. Passing workloads show rel_err up to ~8-9e-03; failing ones crossed over to ~1.1-1.4e-02. This tolerance is strict enough that neither BF16 nor FP16 scores_smem can ship on this dataset.

**Precision analysis (matches theory):**
- BF16: 7 mantissa bits → per-element rounding ~0.78%, accumulated over 64 H-heads: worst case ~sqrt(64) × 0.78% = 6.2%. Observed rel_err up to 1.4e-02, consistent.
- FP16: 10 mantissa bits → per-element rounding ~0.1%, accumulated: worst case ~0.8%. Observed rel_err up to 1.07e-02, slightly above theoretical (probably correlated-error amplification from the MMA accumulation upstream, which isn't a simple random walk).
- FP16's 8× better mantissa cut failures from 16→2 but didn't eliminate them. Going to TF32 wouldn't help (same footprint as FP32).

**Latency mechanism (BF16 vs FP16 disparity):**
- BF16 conversion is essentially free: `__float2bfloat16` ≈ bit truncation of high 16 bits of FP32. Store and load are 2-byte ops.
- FP16 conversion requires exponent rebiasing + mantissa shift: `__float2half` is a non-trivial PTX op (`cvt.rn.f16.f32`). Observed ~10% kernel regression from conversion overhead alone.
- This partially inverts expectations: BF16 is the *faster* lossy variant on B200, but FP16 is the *more accurate* one. Neither hits both targets simultaneously for this kernel.

**Not pursued:** packed `__nv_bfloat162`/`__half2` reads in H-reduction to address the 2-way bank conflict introduced by 2-byte SMEM accesses at stride-65. Given both variants already fail correctness, bank-conflict optimization is moot.

**Implication for M2.3 (FP16 final_scores):** the same tolerance ceiling applies. If scores_smem can't tolerate FP16 precision loss on 16 workloads, final_scores — which is sorted and the top-K boundary examined — will likely fail similarly. Do not pursue M2.3 without first discovering a way to stay within rel_err <1e-2.

**Implication more broadly:** further precision-reducing optimizations are dead-ends for this dataset. Remaining optimization paths must preserve FP32 arithmetic. Candidates still viable: architectural changes (cluster DSM for Q sharing, etc.) and algorithmic (if radix-select can be implemented without widening the error ceiling).

Kernel reverted to 0.039 ms FP32 baseline. Confirmed 2026-04-19 on Modal B200.

### GT-41: Second Sweep (H-reduce ILP + 6 refinements) — Cumulative −2.5% to 0.039 ms

After GT-40's 0.040 ms baseline, an external code-review suggestion ("elect_one_sync serialization + sequential 64-iter H-reduction add latency") triggered a D1-V4 re-analysis. Seven candidates tested:

| # | Candidate | Result | Mean | Notes |
|---|---|---|---|---|
| NEW | H-reduction 4-way / 8-way partial-sum ILP | MARGINAL | 0.040 | Compiler was already extracting ILP from unrolled serial `acc`; 8-way kept for tiny p50/min improvement |
| 1 | Scoring launch_bounds (128,2)→(128,3) | NEUTRAL/slight regression | 0.040 | Register pressure prevented actual occupancy gain |
| 2 | SMEM-cache block_table in top-K | NEUTRAL | 0.040 | L2 already caches block_table warm across kernels; reverted |
| 3 | **Q load via cp.async** | **WIN** | **0.039 (−2.5%)** | Overlapped with K page 0 cp.async; eliminated one __syncthreads |
| 4 | cp.async.ca → cp.async.cg (L1 bypass) | REGRESSION | 0.041 (+5.1%) | Counter to intuition: B200 L1 helps even for one-time loads |
| 5 | Top-K large launch_bounds (512,1)→(512,2) | REGRESSION | 0.043 (+10.3%) | Halving per-thread register budget caused BlockRadixSort spills |
| 6 | L2 persistent cache for block_table | NEUTRAL | 0.039 | Max −2.5% but mean unchanged; reverted for code simplicity |

**Cumulative this session:** 0.040 → **0.039 ms (−2.5%)** vs GT-40 baseline; **−95.1% vs GT-BASELINE** (0.803 ms). Only #3 produced a real win.

**Key findings:**
- **External feedback on H-reduction was partially wrong:** the suggestion predicted 3-8% win from breaking the serial `acc` dependency, but actual gain was <1%. The compiler (nvcc with #pragma unroll 16) was already extracting most of the available FMA-pipeline ILP from the serial chain. Lesson: assumption "compiler can't break serial accumulator dependency" is only true for *arbitrary* code — for simple FMA chains in unrolled loops, modern compilers DO reorder effectively.
- **External feedback on elect_one_sync was incorrect:** tcgen05.mma accumulator RAW dependency forces serialization regardless of how issuance is parallelized. Hardware constraint, not software bottleneck.
- **#3 Q cp.async was the real overlooked win:** the original kernel's synchronous Q prologue had its own __syncthreads. Merging into the page-0 cp.async group eliminated the extra barrier AND let Q load overlap with TMEM alloc + mbarrier init + weights load. Simple change, clean 2.5% win.
- **cp.async.cg surprise regression (#4):** on B200 sm_100, the L1 cache participates in cp.async load paths (line coalescing, prefetch, etc.) even for data that will never be re-read from L1. Bypassing L1 measurably slows GMEM→SMEM transfer. Revise: "L1 is only beneficial for reused data" is FALSE on B200 for cp.async. GT-36 candidate if this re-emerges.
- **Top-K (512, 2) regression (#5):** CUB BlockRadixSort<float, 512, 16> with 16 items/thread holds 16 keys + 16 vals in registers = ~128 registers per thread just for state. Forcing 2 CTAs/SM drops register budget from ~128/thread to ~64/thread, causing spills to local memory (L1-cached, but slow). Lesson: launch_bounds occupancy hint isn't free — register budget impact dominates for register-heavy kernels. Confirmed via regression; did not run `-Xptxas=-v` explicitly but behavior matches.
- **L2 persistent cache (#6) was neutral:** block_table is only B × max_num_pages × 4 bytes (max ~46 KB), naturally resident in L2 between scoring and top-K kernel launches. Explicit persistence policy added nothing measurable. Kept as tool-in-pocket for future kernels where L2 pressure from other data might evict block_table.

Kernel snapshot at `tier3_derivation/kernel_current_baseline.cu`. Cumulative from all sessions: 0.803 → **0.039 ms (−95.1%)**. Confirmed 2026-04-19 on Modal B200.

### GT-40: Seven-Candidate Sweep Against 0.056 ms Baseline — Cumulative −28.6% to 0.040 ms

Sequential empirical test of seven optimization candidates produced by D1-V4 pass over the Opt 2 pipelined kernel. Results (all 128/128 PASS where tested):

| # | Candidate | Result | Mean | vs prev |
|---|---|---|---|---|
| 1 | PAGES_PER_CTA 4→8 | REGRESSION | 0.066 | +17.9% → REVERTED |
| 2 | Custom block radix-select | SKIPPED | — | K/N=0.36 makes partition+K-sort > full sort; ~200 LOC for predicted net loss |
| 3 | Adaptive top-K CTA size (small 256×8=2048 for N_max≤2048) | **WIN** | 0.050 | −10.7% |
| 4 | Fuse fill_-1 into sort kernel (remove torch fill_) | **WIN** | 0.049 | −2.0% |
| 5 | Bank-conflict-free scores_smem (stride 64→65) | **MASSIVE WIN** | 0.041 | **−16.3%** |
| 6 | Vectorized float2 final_scores write | REGRESSION | 0.042 | +2.4% → REVERTED |
| 7 | cp.async K loads (+ remove per-token zero-fill) | WIN | 0.040 | −2.4% |

**Cumulative:** 0.056 → **0.040 ms (−28.6%)** vs GT-35's Opt 2 pipelined baseline; **−95.0% vs GT-BASELINE's 0.803 ms**.

**Key findings:**
- **#1 regression root cause:** doubling PAGES_PER_CTA halves grid_x, cutting SM utilization for small-to-medium batches. Per-CTA work doubles while parallelism halves — same serial wall time, worse SM occupancy. GT-13's min=2 floor is tight but the 4→8 upgrade isn't a free win; the current 4 is already in the sweet spot for this dataset's B distribution.
- **#3's 10.7% win:** small workloads (N_max ≤ 2048, roughly max_num_pages ≤ 32) were wasting sort capacity 4× by using the 8192-cap kernel. Min latency dropped 35% (0.034→0.022 ms). Adaptive dispatch at N_max≤TOPK_SMALL_CAPACITY routes to a 256×8-item CTA.
- **#5's 16.3% win (biggest of session):** GT-30 characterized the bank conflict as "2-way, accepted" — actually 16-way on writes. Each of 16 MMA-active lanes in a warp wrote to the same bank per iteration (row * 64 stride made bank = col%32 for 16 threads). Padding row stride to 65 (co-prime with 32 banks) makes bank = (row + col) % 32, spreading across all 32 banks. The write pattern was the dominant cost — GT-30 missed it because it only analyzed reads.
- **#6 regression root cause:** switching from 64 threads × 1 col to 32 threads × 2 cols (to enable float2 vectorized write) introduced 2-way bank conflict on scores_smem reads (pattern `2*tid + {0,1}` with stride 65 → banks (h+2*tid+c)%32 for tid=0..31 covers only 16 distinct banks, 2-way collision). The write savings didn't cover the read conflict cost.
- **#7's 2.4% win:** cp.async.ca.shared.global for K loads (16-byte async copies direct GMEM→SMEM without register round-trip), plus removing per-token zero-fill. Zero-fill was unnecessary because tcgen05 MMA output cols are independent (output[M,col] depends only on K[:,col]), so garbage K for tokens past seq_len doesn't contaminate valid cols — the final_scores mask at GMEM write stage discards results for invalid cols. p95/max slight regression (+2.6-2.7%) reflects cp.async having overhead on workloads already well-pipelined; mean win reflects the improvement on small-to-medium workloads.

Corrected baselines: GT-30's bank conflict analysis was too forgiving (claimed 2-way accepted; was actually 16-way on writes costing ~16% of kernel time). GT-34's regression prediction was conservative (predicted ~50% for Streaming Gate; actual was 100%). Paper analysis underestimates SMEM bank-conflict-sized effects and overestimates algorithmic-serialization tolerance.

Kernel snapshot at `tier3_derivation/kernel_current_baseline.cu`. Confirmed 2026-04-19 on Modal B200 (seven Modal runs: app ids ThFareMhv0b5clce4kWZNi, 9nzLEKvSB5HjLXAMa2DnRw, 39968VFUiNaxkbvyK4s8pM, oUiFOF6TAd0yxIdUpbZPtq, EjYnNKvaJnPqU8ZxbCNQPZ, TpVJ5dva4lHVTCvBAmaXcZ).

### GT-35: K Double-Buffer Software Pipelining — WIN (−3.4% mean, corrects GT-33 over-pessimism)

Adding a double-buffered K_smem + scale_smem with software pipelining to overlap the next page's K/scale cooperative load with the current page's tcgen05.mma pipeline IS a win, contrary to the GT-33 roofline-based prediction. Confirmed run on Modal B200 (app ap-uD5w58K2ImGvRYyFYGeTD1): mean 0.058→**0.056 ms** (−3.4%), p50 0.050→**0.048 ms** (−4.0%), p95 0.103→**0.101 ms** (−1.9%), min 0.034→0.034 ms (unchanged), max 0.107→**0.103 ms** (−3.7%). 128/128 PASS, abs_err=0.00 rel_err=0.00.

Structure: two K_smem buffers at K0_SMEM_OFFSET=8192 and K1_SMEM_OFFSET=16384 (8192 B each), two scale_smem buffers at SCALE0/SCALE1 (256 B each). Prologue loads page 0 into buffer 0 + __syncthreads. Main loop per pidx: (1) dispatch MMA on K_smem_buf[pidx&1], (2) if has_next, all 128 threads issue cooperative load of page pidx+1's K + scale into K_smem_buf[(pidx+1)&1] (overlaps with MMA since MMA reads curr_buf only), (3) mbarrier_wait on curr MMA, (4) tcgen05.ld + scale + write scores_smem using scale_smem_buf[curr_buf], (5) __syncthreads (makes K_smem_buf[next_buf] writes visible for next iteration's MMA AND scores_smem for H-reduction), (6) SMEM-transpose H-reduction + final_scores write, (7) __syncthreads. SMEM total grows 33328 → 41776 B (+25%), still fits 2 CTAs/SM on B200's 228 KB/SM. `num_active = min(PAGES_PER_CTA, max(0, max_num_pages - cta_page_start))` handles tail CTAs cleanly; loads for pidx+1 guarded by `has_next`.

**Why GT-33's gating was too pessimistic:** GT-33 argued the kernel is memory-bound at ~117 FLOP/byte vs FP8 ridge ~562, so pipelining cannot help. This conflates two different notions of "memory-bound":
1. **AI < ridge** (kernel would be memory-bound IF HBM were saturated) — TRUE.
2. **HBM bus saturated** (HBM controller actually throughput-limited) — FALSE.

The kernel uses ~550 GB/s of B200's 8 TB/s HBM (~7%) at peak. The bottleneck is per-CTA stalls on individual K loads, not HBM controller throughput. Pipelining overlaps those stalls with MMA compute (tcgen05.mma is async to tensor cores; while the MMA hardware pipeline runs, the CTA's 128 threads are free to issue loads). The win is bounded by what fraction of per-CTA time is load-stall (~10-15%), matching the observed −3.4% mean improvement.

**Correction to GT-8:** GT-8's "Pipelining is ABSENT — do not add pipeline infrastructure" rule was correct when the naive kernel had abundant parallelism (B×max_num_pages CTAs, well-saturating SMs) and was truly HBM-bottlenecked by FP32 K materialization. After Tier 3 collapsed K to FP8-end-to-end (GT-28), HBM traffic dropped 4× and per-CTA stall latency became the dominant wait. GT-8's rule stands for FP32-K kernels but does NOT apply to FP8-K kernels with low HBM utilization. Future pipelining attempts should check HBM utilization via NCU before applying; the rule of thumb is: if HBM < 30% utilization at current tier, pipelining may still win.

Kernel snapshot at `tier3_derivation/kernel_opt2_pipelined.cu`. New state baseline: **0.056 ms mean**. Confirmed 2026-04-19 on Modal B200.

### GT-34: Streaming Gate (Single-CTA-Per-Batch Fused) — REGRESSION (+100% mean) AND CORRECTNESS FAILURE

Empirical test of single-CTA-per-batch fused scoring+topk — grid (B,), 512 threads (4 warps MMA-active, all 512 sort-active), full max_num_pages page loop in one CTA with SMEM run_scores accumulation, followed by in-kernel `cub::BlockRadixSort<float, 512, 16, int32_t>` + inline translate, output directly to topk_indices (no final_scores intermediate, no separate top-K kernel). Confirmed run on Modal B200: mean 0.056→**0.112 ms (+100%)** AND **2/128 workloads INCORRECT_NUMERICAL** with abs_err=7.77e+01 on the largest workloads. Double failure — latency and correctness both regressed.

Latency regression by workload class:
- Small (min): 0.034→0.031 ms (−8.8%, the ONLY workloads where single-CTA wins because per-batch parallelism is moot)
- Median: 0.048→0.107 ms (+123%)
- p95: 0.101→0.244 ms (+141%)
- Max: 0.103→0.248 ms (+141%)

Latency mechanism (expected): collapsing G = ceil(max_num_pages/4) ≈ 23 CTAs per batch down to G=1 serializes all pages within a batch. At max_num_pages=90, single-CTA scoring = 90 × ~650 cycles = ~59K cycles ≈ 29 µs per CTA; current multi-CTA design = 5 waves × ~3 µs = ~15 µs. The 2× slowdown on large workloads reflects this serialization loss; small workloads (max_num_pages ≤ 4) naturally run in one CTA anyway and see no per-batch-parallelism penalty, so they saw a tiny improvement from the eliminated launch overhead.

Correctness failure root cause (not debugged, kernel reverted): likely a synchronization issue between warps 0-3 (MMA-active) and warps 4-15 (idle during MMA) in the context of `tcgen05.ld.sync.aligned` + `tcgen05.wait::ld`. GT-17 requires all 32 lanes of the ISSUING warp to participate in tcgen05.ld, which my design satisfies (warps 0-3 all 32 lanes issue). Idle warps 4-15 just skip to __syncthreads. On paper this should be correct, but the empirical 77.7 abs_err on specifically the 2 largest workloads suggests an ordering issue that scales with max_num_pages — possibly interaction between the CUB radix sort's internal __syncthreads and the TMEM hardware state, OR an issue with 512-thread block vs 128-thread-optimized tcgen05 lifecycle. Since the latency regression alone already disqualifies the design, root cause investigation was not pursued.

**Key empirical lesson (corrects GT's paper-only predictions):** The architectural analysis in a previous GT-34 draft predicted ~50% regression. Actual regression was ~100%. The analysis underestimated the per-CTA scoring cost (streaming was slower than 29 µs, closer to 50 µs), and also missed that CUB BlockRadixSort with 512 threads + 16 items/thread has significant SMEM bank conflict overhead when launched from the same CTA that just completed 90 page iterations with different SMEM patterns. Paper roofline models are unreliable for fusion decisions when thread-count trade-offs span mismatched kernel phases (MMA wants 128, radix sort wants 512).

**Architectural conclusion stands:** Streaming Gate at K=2048 with no cluster-DSM is not viable. Any future attempt would need CTA clusters sharing a top-K buffer via distributed shared memory (sm_100a supports up to 8-CTA clusters). That path was not pursued in this optimization cycle.

Kernel snapshot (the failing version) not retained. Revert to Opt 2 (kernel_opt2_pipelined.cu) immediately. Confirmed 2026-04-19 on Modal B200 via failed run.

### GT-33: Post-Tier 2(c) Roofline Re-check — Kernel Still Memory-Bound, Q/K Pipelining Gated Off

Re-evaluation of GT-8's roofline against the current Tier 2(c) kernel state (0.058 ms mean): arithmetic intensity on the worst-case workload (max_num_pages=90, B=32) is **~117 FLOP/byte** (MMA 94 MFLOP + scale/ReLU/weighted-sum 1.4 MFLOP + top-K radix ~650 KFLOP-equiv ≈ 96 MFLOP, over ~824 KB HBM traffic of which K-cache is 760 KB). B200's FP8 tensor-core ridge point (~4.5 PFLOP/s FP8 / ~8 TB/s HBM) is **~562 FLOP/byte**. Kernel is **5× below ridge** — firmly memory-bound, more so than pre-Tier 3 (where FP32 K materialization added extra memory traffic but the FP16/32 ridge was lower; the net effect of Tier 3's FP8-end-to-end is to keep HBM traffic roughly flat while raising the ridge ~2× by moving to FP8 math, so the memory-bound gap widened, not narrowed).

Implication: Q pipelining (double-buffered K SMEM + cp.async stages to overlap K loads with prior-page MMA) cannot produce net latency improvement. Pipelining hides per-CTA load latency, but when HBM is the bus-level bottleneck, adding more outstanding loads can't accelerate total traffic — it just queues requests the HBM controller already can't service faster. The mbarrier+MMA already dispatches tcgen05.mma asynchronously (hardware-pipelined on the tensor core), so compute-with-load overlap is ALREADY happening at the tensor-core-driver level without explicit cp.async infrastructure.

Additional cost of adding pipelining: doubles K_smem (+8 KB), scale_smem (+256 B), occupancy drops from 2 CTAs/SM → 1 CTA/SM below ~48 KB SMEM limit after the added 8 KB. Halving occupancy while saving ~10% per-CTA cycle count is a net loss.

GT-8's original ruling — "Pipelining (async loads to hide latency) is ABSENT — the compute intensity does not justify it. Do not add pipeline infrastructure" — remains correct for all optimization tiers pursued so far. Any future kernel that shifts above ridge (e.g., by substantially reducing HBM traffic via a Streaming Gate + multi-head fusion) would require a fresh roofline re-check before pipelining can be considered. Confirmed 2026-04-19 via paper roofline analysis against Tier 2(c) baseline.

Opt 2 (Q pipelining across pages) SKIPPED per user's gating rule ("only proceed if past ridge point") — kernel remains below ridge.

### GT-32: Multi-CTA K-dim Parallelism with Cross-CTA Merge — REGRESSION (+15.5% mean)

Replacing single-CTA `block_topk_translate_kernel` (GT-31) with a two-pass multi-CTA design — `partial_topk_kernel` (G=2 partitions per batch, each CTA = 256 threads × 16 items = 4096 cap, sorts chunk of `final_scores[b, :]` and emits top-K local positions) + `merge_topk_kernel` (per batch, 512 threads × 8 items = 4096 cap, merges G×K=4096 candidates, emits top-K with inline translate) — is a REGRESSION, not a win. Confirmed run on Modal B200 (app ap-K2JgvkzgtoCw46UF1Jd4gH): mean 0.058→**0.067 ms** (+15.5%), p50 0.050→0.060 ms (+20.0%), p95 0.103→0.113 ms (+9.7%), min 0.034→0.042 ms (+23.5%), max 0.107→0.115 ms (+7.5%). 128/128 PASS correctness (abs_err=0.00 rel_err=0.00) — the regression is purely latency.

Root cause — serial work increase dominates parallelism gain:
1. Radix sort work scales ~linearly with N. Single-CTA sort of 8192 padded elements = 8192 work units. New design: 2 × 4096 (partial, padded) + 1 × 4096 (merge) = 12288 work units = **1.5× MORE radix work** regardless of parallelism.
2. Parallelism gain is bounded because at the problem sizes in this dataset (B up to ~32, 148 SMs), the original single-CTA top-K is ALREADY using all batches × 1 CTA = B CTAs. The SMs for top-K aren't saturated, but each individual CTA's work is highly parallel *within the CTA* already — adding more CTAs doesn't shrink wall time for any one batch's top-K meaningfully because the single CTA's radix sort is already short (~3-5 µs).
3. Extra GMEM round trip: partial_scores[B, G, K] + partial_pos[B, G, K] materialize between the two kernels = extra ~2 MB HBM traffic at B=32. At 8 TB/s, ~0.25 µs overhead per batch.
4. Extra kernel launch overhead: one additional kernel launch (partial + merge = 2 launches vs 1).
5. The min degraded the most (+23.5%) — tiny workloads pay all the extra overhead but gain no parallelism benefit (for N_max << K, the sort is already negligible and the extra launch + GMEM round trip dominate).

Key insight to preserve: the CURRENT single-CTA top-K is optimal for this dataset because BlockRadixSort within one CTA already extracts the maximum intra-CTA parallelism, and for K=2048 the splitting theorem forces each partition to emit K candidates (not K/G), which negates the work-reduction benefit multi-CTA top-K is supposed to provide. **Multi-CTA top-K only beats single-CTA when K << N/G, which this dataset never satisfies** (K=2048, max N=5760 → max usable G=2 → per-partition emission = K, not K/G).

Do not re-attempt multi-CTA top-K with cross-CTA merge at this TOPK/N_max combination. If the dataset ever shifted to K << N (e.g., K=128 with N=5760), the analysis would flip and G could be large with per-partition emission = K/G. Confirmed 2026-04-19 on Modal B200.

### GT-31: Tier 2(c) CUB BlockRadixSort Fused Top-K + Translate — WIN (−26.6% mean vs Option C)

Replacing `at::topk(final_scores, k_req, -1, true, true)` + `translate_indices_batched_kernel` with a single custom `block_topk_translate_kernel` using `cub::BlockRadixSort<float, 512, 16, int32_t>` + `cub::BLOCK_LOAD_WARP_TRANSPOSE` is a dominant Tier 2 win. Confirmed run on Modal B200 (app ap-tWAb7KnfKBCayyB1UweS1F): mean 0.079→**0.058 ms** (−26.6%), p50 0.074→**0.050 ms** (−32.4%), p95 0.139→**0.103 ms** (−25.9%), min 0.034→0.034 ms (unchanged), max 0.144→**0.107 ms** (−25.7%). 128/128 PASS, abs_err=0.00 rel_err=0.00, peak speedup vs Python reference **156.98×** (up from 143.77×).

Structure: one CTA per batch, `(B,)` grid, 512 threads × 16 items/thread = 8192 capacity (covers all dataset N_max up to 5760 with headroom). Per CTA: STRIPED coalesced `BlockLoad` from `final_scores[b, :N_max]` (padding with `-INFINITY` beyond N_max) producing BLOCKED arrangement → `BlockRadixSort::SortDescending(keys, vals)` (full block sort, 8× 4-bit radix passes on 32-bit FP32) → BLOCKED output with thread t holding ranks `[t*16, (t+1)*16)` → per-thread top-K extraction with inline translate `phys = block_table[b, lid>>6]; topk_indices[b, rank] = phys*64 + (lid&63)`. `topk_indices.fill_(-1)` pre-stamp handles `rank ≥ actual_topk` positions (left as -1).

Mechanism of the win (larger than the 5-15% D3 projection):
1. Eliminates the separate int64 `local_idx [B, k_req]` tensor allocation inside `at::topk` — the 2 KB per batch × B × 8 B = bytes add up, but more importantly the `at::topk` API materializes both `values` AND `indices`, and we only used indices (values was wasted HBM bandwidth).
2. Eliminates the `translate_indices_batched_kernel` launch (~1-2 µs launch overhead × B internal dispatch).
3. Fuses translation into the top-K output write — no intermediate int64 → int32 narrowing step.
4. `at::topk` evidently carries more driver overhead than a single custom kernel launch; PyTorch's TopK dispatch path seems to have per-batch overhead that a single fused kernel avoids.

D3 projection was 5-15% improvement; actual was **26.6%**. Post-hoc explanation: at::topk's wasted `values` output HBM bandwidth + launch sequencing overhead were both larger than I estimated.

The full sort (O(N log N) via 8 radix passes) costs more theoretically than a pure radix-select (O(N)) but the K/N ratio in this dataset (0.36-1.0, median ~0.8) makes the saved work from early termination marginal. CUB's BlockRadixSort is highly optimized and avoids the selection-state tracking overhead.

Dependencies: `<cub/block/block_radix_sort.cuh>` + `<cub/block/block_load.cuh>` from CUDA 12.8's CUB. No external packages.

This is the new state baseline. Any future optimization is measured against Tier 2(c) mean = **0.058 ms** (−92.8% from GT-BASELINE). Confirmed 2026-04-19 on Modal B200. Kernel snapshot at `tier3_derivation/kernel_tier2c.cu`.

### GT-30: SMEM-Transpose H-Reduction Fusion — Option C WIN (−7.1% mean vs Tier 3)

Fusing `weighted_relu_sum` into `fp8_mma_final_scores_kernel` DOES work — the right algorithm is SMEM-transpose, not shuffle-tree. Per-page, after TMEM→register scale-multiply, each live lane writes its 64 scaled-score values into the 16 KiB `scores_smem[64×64]` tile (unchanged from Tier 3). Then `__syncthreads`, then 64 threads (one per N column, `tid < PAGE_SIZE`) each read DOWN their column with a 64-iter FMA chain `acc = fmaf(fmaxf(scores_smem[h*64+col], 0), weights_smem[h], acc)` and write `final_scores[b, n_global]` (stamping -INF for `n_global >= seq_len_b`). The H-reduction has no cross-thread dependency and runs as a plain 64-FMA chain, well-pipelined.

Confirmed run on Modal B200 (app ap-vY54j6IQU50VaxiSHEHq9X): mean 0.085→**0.079 ms** (−7.1%), p50 0.079→0.074 ms (−6.3%), p95 0.146→0.139 ms (−4.8%), min 0.037→0.034 ms (−8.1%), max 0.151→0.144 ms (−4.6%). 128/128 PASS, abs_err=0.00 rel_err=0.00, peak speedup vs Python reference **143.77×** (up from 122×).

Mechanism: eliminates (1) the B×64×N_max FP32 `scores_all` GMEM round trip (~47 MB at max workload, ~47 µs of HBM traffic), (2) the `weighted_relu_sum_batched_kernel` launch overhead, (3) the separate `torch::empty({B, 64, N_max})` allocation. Cost added: 64 threads × 64 SMEM reads × 1 FMA per page ≈ 200-300 cycles per page (2-way SMEM bank conflict on stride-64 reads, accepted — still cheap), one extra 256 B `weights_smem` per CTA. Net: small but positive on all workloads (consistent across small/medium/large).

Key contrast with GT-29 (the Tier 4(b) shuffle-tree attempt): the shuffle-tree does `relu*weight*sum` INSIDE each lane via a 64-column × 5-stage butterfly, creating a ~1600-cycle serial critical path per warp per page that dominated for large `max_num_pages`. Option C moves the reduction into SMEM with no cross-thread reductions at all (each thread's column is independent), which is what unlocks the win. GT-29's "do NOT use per-column shuffle-tree" guidance stands; GT-30 supersedes GT-29's suggested alternative — SMEM-transpose is now confirmed.

This is the new state baseline. Any future optimization is measured against Option C mean = 0.079 ms. Confirmed 2026-04-19 on Modal B200. Kernel snapshot at `tier3_derivation/kernel_optionC.cu`.

### GT-29: Do NOT Fuse H-Reduction (weighted_relu_sum) Inside fp8_mma_scores_kernel — Tier 4(b) REGRESSION (per-column shuffle-tree shape only)

Moving the `weighted_relu_sum_batched_kernel` work (per-column `Σ_h relu(scale*score[h,n])*weights[h]`) INSIDE `fp8_mma_scores_kernel` — replacing the `scores_all[B,64,N_max]` FP32 GMEM round trip with an in-kernel per-column shuffle-tree reduction — is a REGRESSION, not a win. Confirmed run on Modal B200 (app ap-ETQnIfBH4TrpfsaceuC13H): mean 0.085→**0.104 ms** (+22.4%), p50 0.079→0.088 ms (+11.4%), p95 0.146→**0.217 ms** (+48.6%), max 0.151→**0.221 ms** (+46.4%), min unchanged at 0.037 ms. 128/128 PASS correctness (abs_err=0.00 rel_err=0.00) — the regression is purely latency.

Failed design (do not re-try this shape): one `__shfl_xor_sync` butterfly reduction PER column (64 columns × 5 XOR stages × 4 warps), followed by a cross-warp partial sum in SMEM and a 64-thread GMEM write. The per-column shuffle chain creates a serial dependency on the reduction result before the SMEM store can dispatch; the compiler cannot interleave enough across the 64 iterations to hide the ~5-cycle shuffle latency. Estimated critical path ≈ 1600 cycles per warp per page × ~90 pages on large workloads ≈ 90 µs of in-kernel reduction, which exceeds the ~47 µs of GMEM round-trip traffic the fusion was meant to eliminate. Small workloads (`max_num_pages=1`) are unchanged because the reduction cost is trivial there; the regression scales with `max_num_pages` — p95/max degraded the most, exactly where the fusion should have helped.

The regression is not a correctness issue (bit-identical to Tier 3 output) and not a register/occupancy issue (SMEM dropped 33→18 KiB so occupancy should have INCREASED). Root cause is purely the serial shuffle-tree critical path.

Viable-but-untested alternative design (do NOT combine with any other change; evaluate as its own optimization if pursued later): SMEM-transpose H-reduction. Each lane writes its 64 scaled-score FP32 values into an SMEM `[64 rows × 64 cols]` tile, then 64 threads (one per column) read down the column and compute the weighted relu sum in a 64-iter FMA chain. Cost: 16 KiB SMEM tile (same as Tier 3's scores_smem, so no occupancy change), ~64 FMA cycles per column per page, well-pipelined. Not tested here because GT-TIER-SUMMARY + GT-28 targets are met; log as "Option C" for future work.

Do not attempt per-column shuffle-tree H-reduction fusion again. Confirmed 2026-04-19 on Modal B200 via full 128-workload run. Reverted to Tier 3 state immediately.

### GT-28: Tier 3 FP8 tcgen05 Fused MMA Kernel — Cumulative State (2026-04-19)

Replacing `dequant_gather_batched_kernel` + `torch::bmm(q_fp32, K_deq.T)` with a single fused `fp8_mma_scores_kernel` using `tcgen05.mma.cta_group::1.kind::f8f6f4` is a dominant Tier 3 win. Confirmed run on Modal B200 (app ap-Blgrm5PGsA7yzhzo9HlxRc): mean 0.112→0.085 ms (−24.1% vs Tier 2, −89.4% vs GT-BASELINE), p50 0.099→0.079 ms (−20.2%), p95 0.246→0.146 ms (−40.7%), min 0.048→0.037 ms (−22.9%), max 0.259→0.151 ms (−41.7%). 128/128 PASS, abs_err=0.00 rel_err=0.00, per-workload speedup vs Python reference 17–122×.

Structure (inside `fp8_mma_scores_kernel`): grid `(ceil(max_num_pages/PAGES_PER_CTA=4), B)`, block 128 threads (4 warps). Per CTA: TMEM alloc 64 cols (warp 1), block_table cache (warp 0 t0), mbar init (warp 0 t0), Q cooperative load once into 8×T core-tile SMEM (GT-11 SBO=256 LBO=128, swizzle=NONE per GT-10), then per-page loop (≤ 4 iters): K cooperative load with per-token zero-fill mask + scale gather → `__syncthreads` → 4 slab `tcgen05.mma` issued by warp 0 elected thread (no `#pragma unroll` per GT-14) → `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64` (GT-18) → mbar phase-parity wait → `tcgen05.fence::after_thread_sync` → 2× `tcgen05.ld.sync.aligned.32x32b.x32.b32` on all 32 lanes per warp (GT-17) → `tcgen05.wait::ld` → per-column scale multiply in registers (lane<16 mask AFTER load) → scores_smem → coalesced GMEM store → `__syncthreads`. TMEM dealloc on single exit path (warp 1, no early returns below alloc).

IDESC = `0x04100010` = bit4(F32 dtype) | (8<<17)(N=64) | (4<<24)(M=64), all other bits 0 (E4M3 × E4M3, no transpose, no sparsity, no .ws). Scale applied post-MMA pre-relu (plain distributive law — no sign assumption on scale_tok). K stays FP8 end-to-end (4× HBM traffic savings vs Tier 2's FP32 K_deq round trip). PAGES_PER_CTA=4 amortizes Q load across 4 pages (4× Q HBM savings). SMEM budget 33 KiB/CTA, registers ~74/thread → ~6 CTAs/SM.

This is the new state baseline. Any future optimization is measured against Tier 3 mean=0.085 ms. Remaining cost centers: `at::topk` on [B, N_max] (now the largest single kernel cost in the launch_topk_c pipeline), `weighted_relu_sum_batched_kernel`, and the post-MMA FP32 stores to `scores_all`. Next Tier 2/Tier 3 candidates per D1's re-evaluation note: Streaming Gate fusion of top-K into the MMA loop (K/N < 0.5 for `max_num_pages ≥ 65`), or replacing `at::topk` with a custom warp/block radix-select.

Confirmed 2026-04-19 on Modal B200.

---

## Section 8 — Failure Pattern Signatures

Match symptom to class. Go directly to the first action — do not perform a general diagnosis first.

| Symptom | Most likely class | First action |
|---|---|---|
| Wrong output, -1 padding missing or in wrong positions | FP8_DEQUANT or BLOCK_TABLE | Run Python reference run() on smallest failing workload (batch=1, max_num_pages=1). Diff output tensor element by element. |
| Correct on small workloads (batch<=2), wrong on large (batch>=8) | SEQ_LEN | Check actual_topk = min(TOPK, seq_len) is applied. Check seq_len <= 0 short-circuit. |
| COMPILE_ERROR on all 128 workloads, no error text in terminal | MODAL_IMAGE | Check Modal image uses nvidia/cuda devel base (GT-19). Check pack_solution.py binding patch (GT-20). |
| COMPILE_ERROR with binding=None in solution.json | PACK_SOLUTION | Apply GT-20 patch to pack_solution.py. Verify solution.json shows binding: torch after packing. |
| XID 13 on cp.async.bulk.tensor | TENSOR_MAP | Check all globalStrides / 16. K-cache stride is 132 — cannot use TMA (GT-9). |
| XID 13 on tcgen05.mma | TMEM_ADDR | Check TMEM alloc lifecycle (GT-2, Tier 3 section). |
| XID 13 on kernel launch | SMEM | Check dynamic SMEM opt-in > 48KB. |
| Wrong scores, no fault, ~0-20% of reference | SMEM_LAYOUT | Layout fill logic wrong — data reaching MMA is incorrect. |
| Wrong scores, no fault, ~50-80% of reference | DESCRIPTOR | SBO/LBO stride mismatch. Run sbo_lbo_sweep immediately. |
| Wrong scores, no fault, ~100% magnitude but wrong positions | IDESC | transpose_A or transpose_B bit wrong in IDESC. |
| Wrong scores only after adding swizzle | GT-10 violation | TMA swizzle mode does not match SMEM descriptor mode (GT-10, Tier 3 section). |
| ptxas parse error near : | PTX_SYNTAX | tcgen05.fence has illegal suffix — bare form only (GT-7, Tier 3 section). |
| ptxas instruction not supported | PTX_SYNTAX | WGMMA on sm_100a — does not exist (GT-1, Tier 3 section). |
| ptxas "tcgen05.* not supported on .target 'sm_100'" | ARCH_TARGET | `TORCH_CUDA_ARCH_LIST` unset or set to `10.0`. Set to `10.0a` in Modal image `.env({...})` per GT-27. |
| Kernel hangs, no XID, no output | BARRIER | Missing __syncthreads() or infinite loop in sort/reduction. |
| Numerically correct on small inputs, wrong on large | SMEM_OVERFLOW | SMEM buffer overflow — T_pad calculation wrong. |
| Kernel hangs at tcgen05.ld, wrong results on lane-gated warps | TMEM_LD | Remove if(lane<16) guard before tcgen05.ld — all 32 lanes must issue (GT-17, Tier 3 section). |
| RUNTIME_ERROR: invalid resource handle | HARDWARE_MISMATCH | Kernel compiled for sm_100a running on non-B200 GPU. Always test on Modal B200, never local. |

---

## Section 9 — pack_solution.py Contract

**The agent edits exactly one file: `solution/cuda/kernel.cu`.**

`run_modal.py` calls `pack_solution.py` which reads `config.toml` to determine the entry point function name and language, then packages `kernel.cu` into a `Solution` object for submission to Modal.

**Rules that must hold at all times:**

1. `config.toml` field `entry_point` must exactly match the function name in `kernel.cu`. Current value: `kernel.cu::launch_topk_c`. If this function is renamed during optimization, `config.toml` must be updated in the same step — not after.

2. `config.toml` field `definition` must remain `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`. Never change this.

3. `pack_solution.py` must have the GT-20 binding patch applied. Without it, `binding` defaults to tvm-ffi and compilation fails. Verify by running `python scripts/pack_solution.py` and checking that `solution.json` shows `"binding": "torch"`.

4. If additional `.cu` or `.h` files are added during optimization, verify `pack_solution.py` picks them up by inspecting what it packs before running `run_modal.py`. A mismatch causes silent wrong-function compilation — the symptom looks like a correctness failure but the actual kernel never ran.

5. Current `config.toml` state (do not modify definition or binding):
```toml
[solution]
name = "KarnbirKhera"
definition = "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64"
author = "KarnbirKhera"

[build]
language = "cuda"
entry_point = "kernel.cu::launch_topk_c"
binding = "torch"
destination_passing_style = true
```

---

## Section 10 — Phase 2: Optimization Loop

Phase 2 begins now. `checkpoints/kernel_naive.cu` is locked.

### Per-optimization cycle structure

```
1. Scope declaration
2. Pre-implementation analysis (tiered — see below)
3. Implement in kernel.cu
4. Update config.toml if entry_point changed
5. modal run scripts/run_modal.py
6. Read output
   -> PASS 128/128: keep change, record timing vs GT-BASELINE, apply GT Update Rule if new fact found
   -> FAIL: revert kernel.cu to checkpoint immediately, write GT entry for what broke
7. Proceed to next optimization
```

**Scope declaration (mandatory before every optimization):**
Write three sentences: (a) what this optimization changes, (b) what it explicitly does not change, (c) which checkpoint is locked for revert.

**One optimization at a time.** Never combine two independent changes in one test cycle. If two changes are needed together for correctness, that counts as one optimization — document why they are coupled.

**Revert command:**
```bash
cp ~/CUDAExperiments/ClaudeCode3.0/checkpoints/kernel_naive.cu \
   ~/CUDAExperiments/ClaudeCode3.0/flashinfer-bench-starter-kit/solution/cuda/kernel.cu
```

### Tier 1 — Free Wins (Sonnet, lightweight checklist)

Applies to: coalescing improvements, vectorized loads, warp utilization, dead code removal, redundant barriers, compiler hints.

No full D1-D4 required. Before implementing, answer these three questions in writing:

- Does this change affect the memory access stride of any tensor? If yes -> run D3 scoped to the affected access pattern before implementing.
- Does this change affect any barrier placement? If yes -> run D3 scoped to the barrier before implementing.
- Does this change touch the FP8 dequant path? If yes -> run D3 scoped to the dequant before implementing.

If all three answers are no -> implement directly.

**Tier 1 candidates for this kernel (evaluate in order):**
1. The batch loop is serial on CPU — each batch element launches kernels sequentially. For small batch sizes this is fine; for large batch sizes (batch=16+) this serializes GPU launches. Evaluate batching multiple elements per launch.
2. `dequant_gather_kernel` launches with grid=(num_pages_seq, PAGE_SIZE), block=32. Each block handles one (page, token) pair. Evaluate fusing page dimension to improve occupancy.
3. `sl_cpu = seq_lens.to(torch::kCPU)` synchronizes the GPU every launch. Evaluate pinned memory or keeping seq_lens on CPU from the start.
4. `K_deq_buf` is allocated fresh every launch. Evaluate persistent allocation (GT-16 pattern).

### Tier 2 — Algorithm Class (Sonnet, D3 only)

Applies to: top-k algorithm selection (sort vs selection), Streaming Gate implementation, per-CTA buffer architecture, head reduction primitive.

**K/N ratio analysis — already computed:**
K = 2048 (TOPK), N = max_num_pages x PAGE_SIZE. For the largest workloads (max_num_pages=90): N = 5760, K/N ~= 0.356. For median workloads (max_num_pages=32): N = 2048, K/N = 1.0. For small workloads (max_num_pages=2): N = 128, K/N >> 1 (seq_len < TOPK, actual_topk = seq_len).

**Algorithm class decision:** Current implementation uses `at::topk` (library sort). For workloads where K/N < 0.5, selection-based algorithms (radix-select, warp-level insertion buffer) are O(N) vs O(N log N). Evaluate this at Tier 2.

**Streaming Gate:** Scores are computed one page at a time in the page loop (FXP x GATE fires). A streaming top-k buffer that maintains a running top-K during the page loop eliminates the need to materialize all scores before selection. This is the highest-value Tier 2 optimization. FORBIDDEN architecture: writing all N raw scores to global memory before selection — this is the exact round-trip the Streaming Gate eliminates.

Run D3 scoped to the Gate molecule change. D3 item 12 (algorithm selection) is the mandatory decision point.

### Tier 3 — Hardware-Specific (Opus, full D1-D4)

Applies to: tcgen05 MMA for FP8 q@K.T scoring, TMA for Q loads, TMEM accumulator, async pipeline introduction.

**Before starting Tier 3:**
1. Load `ptx_isa_sections/MANIFEST.md` — identifies which ISA files to load for each pipeline step. Load ONLY the sections needed for the current step.
2. Load `gau-nernst_reference.h` — confirmed working B200 PTX wrappers. Cross-check all inline PTX against this.
3. Read Section 11 (Tier 3 GT entries) in full.
4. Run full D1->V1->D2->V2->D3->V3->D4->V4 scoped to the hardware change.

Model routing for Tier 3: D3, V3, D4, V4, Layer 4 implementation, and Diagnostic Agent -> **Opus**. All other steps -> Sonnet.

### Tier 4 — Micro-Optimizations (Sonnet, no D1-D4)

Applies to: register pressure tuning, instruction scheduling, shared memory padding, loop unrolling decisions.

Apply one micro-optimization at a time. Test after each. No derivation pipeline needed — these are local code changes with no structural impact.

### Optimization re-evaluation rule (mandatory after each tier)

After completing a tier, profile the kernel and identify the dominant cost center. Before proceeding to the next tier, ask: "Does the dominant cost center correspond to a molecule whose algorithm class might be wrong?" If the dominant cost is the top-k selection phase and K/N is below 0.5, selection-based alternatives must be tested before declaring Tier 2 complete. Do not proceed to Tier 3 while a Tier 2 algorithmic improvement remains untested.

---

## Section 11 — Tier 3 Only: Hardware GT Entries

**DO NOT READ THIS SECTION until Tier 3 is the active optimization target.**
These entries are irrelevant noise during Tiers 1-2. Reading them early risks premature use of tcgen05 instructions.

---

### GT-1: WGMMA Does Not Exist on sm_100a

`wgmma.mma_async`, `wgmma.fence`, `wgmma.commit_group`, `wgmma.wait_group` are Hopper-only (sm_90a) and DO NOT COMPILE on sm_100a. WGMMA fragment indexing formulas must never appear in tcgen05 kernels.

### GT-2: tcgen05 Instruction Family

- **Compute:** `tcgen05.mma.cta_group::1.kind::f8f6f4` for FP8. Look up valid shapes in PTX ISA Table 41 (S9.7.16.2.1), idesc in Table 44 (S9.7.16.4.2) at implementation time.
- **Accumulator:** TMEM (dedicated per-SM memory), NOT registers. Transparent row-major layout — no fragment indexing.
- **Issuing:** Single elected thread via `elect_one_sync()`.
- **TMEM lifecycle:**
  1. tcgen05.alloc (full warp, warp_id==1)
  2. MMA (elected thread issues)
  3. tcgen05.commit (elected thread, immediately after MMA)
  4. mbarrier.try_wait (consumer threads — warps 0 and 1 — spin until MMA hardware done)
  5. tcgen05.fence::after_thread_sync (consumer threads, after mbarrier wait)
  6. tcgen05.ld -> tcgen05.wait::ld
  7. tcgen05.dealloc before kernel exit (all paths)
- **TMEM quarters:** For M=64, warps 0-1 hold rows 0-63.

### GT-3: tcgen05.wait::mma Does Not Exist

PTX ISA S9.7.16.8.5: `.wait_operation = { .wait::ld, .wait::st }` only. MMA->ld ordering uses `commit -> mbarrier_wait -> fence::after_thread_sync` (GT-2 steps 3-5). tcgen05.fence::before/after_thread_sync around __syncthreads() is for cross-thread ordering of pipelined instruction pairs only — it does NOT wait for MMA hardware pipeline completion.

### GT-4: H-Reduction Pattern

Fence+sync+fence -> TMEM load -> per-slab scale in registers -> intra-warp __shfl_xor_sync across 32 heads -> two-warp cross-SMEM merge. h_partial_smem sized for **2 active warps** (not 4).

### GT-5: Scale Application

Per-slab MMA with enable_input_d=false (fresh accumulator) -> fence+sync+fence -> tcgen05.ld + wait::ld -> scale multiply in registers -> accumulate into running register total. Never inside MMA pipeline.

### GT-6: TMA Tensor Map Constraints

- **Residency (S9.7.9.27.1.2):** tensorMap for cp.async.bulk.tensor MUST be in .param/.const/.global — NOT .shared.
- **Modification flow:**
  1. Lane 0: memcpy template -> SMEM, tensormap.replace in SMEM
  2. __syncwarp() for visibility
  3. All 32 lanes: tensormap.cp_fenceproxy SMEM -> global (.sync.aligned)
  4. Thread 0: fence.proxy.tensormap::generic.acquire.cta on global copy
  5. Thread 0: TMA using global copy's address
- **tensormap.replace syntax:** requires BOTH .b1024 AND .b64 qualifiers.
- mbarrier expect-tx: call once with total transfer size.

### GT-7: PTX Syntax Pitfalls

- **tcgen05.fence::before_thread_sync** — bare form only. Do NOT append ::1.cta.sync.aligned.
- **Avoid <cuda/ptx> header entirely** — cuda::ptx:: wrappers have __CUDA_ARCH__ guard issues on sm_100a. Use raw asm volatile(...).
- **<cuda/barrier> is fine** to include.

### GT-10: TMA/MMA Swizzle Consistency

If TMA loads data with CU_TENSOR_MAP_SWIZZLE_128B, bytes are rearranged in SMEM. The MMA SMEM descriptor built by make_smem_desc (bits[46:48]=0b001) assumes LINEAR layout. These are INCOMPATIBLE — MMA reads scrambled data, produces wrong results silently (no fault). Fix: use CU_TENSOR_MAP_SWIZZLE_NONE for all TMA loads feeding into tcgen05.mma via linear make_smem_desc. Confirmed 2026-04-12 on Modal B200.

### GT-11: tcgen05.mma SMEM Descriptor — SBO/LBO Confirmed Values

For tcgen05.mma.cta_group::1.kind::f8f6f4, M=64/N=64/K=32, K-major, SWIZZLE_NONE, sm_100a, using the 8xT core-tile SMEM layout:

    SBO field bits[32:45] = 16   (SBO = 256 bytes)
    LBO field bits[16:29] = 8    (LBO = 128 bytes)

Confirmed 2026-04-15 by host-side sweep on Modal B200 (diagnostic_tests/sbo_lbo_sweep.cu).

**SMEM layout constraint:** Q and K SMEM tiles must use the 8xT core-tile layout. Fill formula:
`addr = m_grp * SBO_bytes + k_tile * LBO_bytes + m_in_grp * 16 + k_in_t`
where m_grp = m/8, m_in_grp = m%8, k_tile = k/16, k_in_t = k%16.

### GT-12: tcgen05.ld Lane Distribution for M=64

For kind::f8f6f4 M=64 cta_group::1, the 64 M-rows are distributed 16 per warp across all 4 warps. Only lanes 0..15 of each warp hold live MMA output data (lanes 16..31 read zeros). Canonical tcgen05.ld address for accumulator row r: `(warp_id * 32 + lane_id) << 16` where lane L < 16 of warp W maps to M-row (W * 16 + L). Upper lanes must be masked AFTER the load (not before — see GT-17).

### GT-14: Slab Loop Unrolling Breaks MMA Ordering

#pragma unroll on the K-slab loop breaks MMA->ld ordering and produces wrong results. Do not unroll the slab loop. Confirmed 2026-04-15 on Modal B200.

### GT-15: IDESC Transpose B Bit with K-Major 8xT Layout

For the K-major 8xT core-tile SMEM layout (GT-11), the IDESC transpose_B bit must be 0, not 1. transpose_B=1 produces the canonical 50-80% descriptor-mismatch signature. Confirmed 2026-04-15 on Modal B200.

### GT-16: Persistent Global Scratch for Multi-CTA Kernels

For multi-CTA kernels requiring inter-kernel scratch buffers, use a static device pointer (allocated once, never freed in-process). Per-launch cudaMalloc/cudaFree eliminates parallelism gains from multi-CTA partitioning. Confirmed 2026-04-15 on Modal B200.

### GT-17: tcgen05.ld Requires Full Warp Participation

tcgen05.ld.sync.aligned requires ALL 32 lanes of the issuing warp to execute the instruction. Gating with if(lane_id < 16) before tcgen05.ld causes hang or wrong results. Correct pattern: all 32 lanes issue tcgen05.ld; downstream computation is masked by if(lane_id < 16) AFTER the load returns. Confirmed 2026-04-17 on Modal B200.

### GT-18: tcgen05.commit Correct Form for CUDA 12.8

`tcgen05.commit.cta_group::1.mbarrier::complete_tx::bytes` does NOT compile under CUDA 12.8. Correct form:

    tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [addr];

Always use .mbarrier::arrive::one.shared::cluster.b64. Confirmed 2026-04-17 on Modal B200 CUDA 12.8.

---

## Section 12 — Testing Protocol

**The single canonical test command:**

```bash
cd ~/CUDAExperiments/ClaudeCode3.0/flashinfer-bench-starter-kit
source ~/CUDAExperiments/ClaudeCode3.0/fi-bench-env/bin/activate
modal run scripts/run_modal.py
```

**Reading the output:**
- `PASS: all workloads passed.` -> 128/128 passed
- `FAIL: N workloads failed` -> N workloads failed, per-workload detail above it
- `PASSED: X/128` in the summary header -> X passed out of 128

**Retry limit:** 3 retries per attempt before escalating to Diagnostic Agent.

**Timing output:** The script prints mean, p50, p95, min, max latency across all workloads. Record these after every passing optimization and compare against GT-BASELINE.

**Local smoke test (compile check only, NOT a correctness test):**
```bash
cd ~/CUDAExperiments/ClaudeCode3.0/flashinfer-bench-starter-kit
export FIB_DATASET_PATH=~/CUDAExperiments/ClaudeCode3.0/mlsys26-contest
python scripts/run_local.py
```
Local gives RUNTIME_ERROR (RTX 4060 cannot run sm_100a code) but will give COMPILE_ERROR if the kernel fails to compile. Use local as a fast pre-check before spending Modal compute.

**Invariants that must always hold:**
- `checkpoints/kernel_naive.cu` is never modified
- `config.toml` definition is never changed
- `config.toml` entry_point always matches the function name in `kernel.cu`
- Tests always run on Modal B200 for final validation
- Local run used only as a compile check

---

## Section 13 — 3-Stage Diagnostic Escalation

Trigger when: an optimization is BLOCKED after 3 retries.

### Stage 1: Diagnostic Agent

Invoke a fresh **claude-opus** agent. Provide:
- Last 3 failure outputs from `modal run scripts/run_modal.py`
- Current `kernel.cu`
- The failure pattern from Section 8 that best matches the symptom
- For Tier 3 failures: `d3.md`, `d4.md`, relevant PTX ISA sections, `gau-nernst_reference.h`

The agent checks the Failure Pattern Signatures table first, classifies the failure, then produces `diagnosis.md` with: FAILURE_CLASS, ROOT_CAUSE, EVIDENCE, AFFECTED_DECISIONS, PROPOSED_FIX, NEW_GT_CANDIDATE.

If FAILURE_CLASS = DESCRIPTOR, the agent runs `modal run run_diagnostic.py` (sbo_lbo_sweep) before proposing any fix.

Apply the proposed fix. Retry 3 times.

### Stage 2: Bisection Agent

If Stage 1 fix fails, invoke a **claude-sonnet** bisection agent that:
1. Starts from `checkpoints/kernel_naive.cu` (last known passing state)
2. Adds one element at a time from the failing kernel
3. Tests each addition via `modal run scripts/run_modal.py`
4. Produces `minimal_repro.cu` (smallest failing file) and `minimal_pass.cu`
5. Reports which specific element causes the failure

### Stage 3: Human Review

If Stage 2 cannot isolate the failure, escalate to human with:
- Bisection report
- `minimal_repro.cu` and `minimal_pass.cu`
- Recommended spec changes with options
- Which D-step files need updating

### GT Update Rule on Resolution

Every successful diagnosis that reveals a new hardware constraint becomes a new GT entry in Section 7 (always-active) or Section 11 (Tier 3 only). Probe outputs that confirm hardware-specific behavior are recorded with date and probe name. This is mandatory — not optional.

---

## Section 14 — Model Routing Table

| Work | Model | Reason |
|---|---|---|
| Tier 1 optimization | Sonnet | Checklist-guided, mechanical |
| Tier 2 D3 (algorithm selection) | Sonnet | K/N ratio is mechanical once computed |
| Tier 3 D3, V3 | **Opus** | Hardware binding — unconstrained reasoning |
| Tier 3 D4, V4 | **Opus** | Architecture spec — must catch D3 errors |
| Tier 3 Layer 4 (IDESC/SMEM descriptors) | **Opus** | Bit-field encoding, TMEM addressing |
| Tier 3 Layers 0-3, 5-7 | Sonnet | Boilerplate and algorithmic code |
| Tier 4 micro-optimizations | Sonnet | Local changes, no structural reasoning |
| Stage 1 Diagnostic Agent | **Opus** | Must reason outside GT rules |
| Stage 2 Bisection Agent | Sonnet | Mechanical: add one thing, test, repeat |

---

## Section 15 — The Mandate

- **"Could", "optionally", "consider", "might" are forbidden.** Every decision is committed. Use "BOUND:", "REQUIRED:", "CONFIRMED:".
- **No tcgen05 instructions before Tier 3 is the active optimization target.** Writing tcgen05 code in Tiers 1-2 is an error.
- **The checkpoint file is never modified after Phase 1 locks it.** Not for debugging. Not for "just trying something." Never.
- **One optimization at a time.** Never combine two independent changes in one test cycle. If a test fails, the cause must be unambiguous.
- **No code in D1-D4 steps.** Derivation steps produce specification documents. Implementation begins only after V4 passes.
- **Save intermediate work immediately.** After each D or V step, save the output file. After each passing test, record the timing numbers vs GT-BASELINE.
- **Never write PTX from memory.** Stop and look it up from the PTX ISA sections first.
- **Local run is a compile check only.** RUNTIME_ERROR locally is expected and harmless. Only Modal B200 results determine pass/fail.