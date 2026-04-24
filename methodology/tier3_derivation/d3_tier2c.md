# D3 — Tier 2(c) Custom Block Radix-Sort for Top-K (algorithm selection only)

## Scope declaration

**(a) What this optimization changes:** Replaces `at::topk(final_scores, k_req, -1, true, true)` + `translate_indices_batched_kernel` with a single custom kernel `block_topk_translate_kernel` that, per batch, block-sorts `final_scores[b, :N_max]` descending via `cub::BlockRadixSort<float, 512, 16, int32_t>` and writes **translated** global token indices directly into `topk_indices[b, :actual_topk]`. Eliminates: (1) `at::topk`'s FP32 values output allocation (kept internally by PyTorch, unused by downstream), (2) the int64 `local_idx` intermediate tensor, (3) the separate `translate_indices_batched_kernel` launch.

**(b) What this optimization does NOT change:** `fp8_mma_final_scores_kernel` (Option C), `final_scores[B, N_max]` format, `topk_indices.fill_(-1)` pre-stamp, `launch_topk_c` signature, `config.toml` entry point.

**(c) Checkpoint locked:** `tier3_derivation/kernel_optionC.cu` (new 0.079 ms mean baseline). `checkpoints/kernel_naive.cu` remains the Phase 1 baseline. Revert both Option C + radix-select changes together if (c) regresses.

---

## D3 item 12 — algorithm selection (mandatory Tier 2 decision)

**K/N ratio distribution across the 128 workloads:**
| `max_num_pages` range | N_max range | actual_topk | K/N | Workload count |
|---|---|---|---|---|
| 1-2 | 64-128 | = N_max | >1 (saturated) | ~10 |
| 3-31 | 192-1984 | = N_max | >1 or near 1 | ~30 |
| 32 | 2048 | 2048 | 1.0 | ~20 |
| 33-64 | 2112-4096 | 2048 | 0.97-0.5 | ~40 |
| 65-90 | 4160-5760 | 2048 | 0.49-0.36 | ~28 |

For `max_num_pages ≤ 32` (roughly half the workloads), **K ≥ N** so `actual_topk = N_max` — no selection needed, just sort (or even just return indices 0..N-1 in score-descending order).

For `max_num_pages ∈ [65, 90]` (~22% of workloads), K/N < 0.5 — selection-based algorithms become theoretically 2× faster than full sort.

**Candidate algorithms:**
| Algorithm | Complexity | Block/grid | Notes |
|---|---|---|---|
| `at::topk` (current) | Radix-select O(N), CUB-backed | Device-level, ~B kernels internally | Proven, but adds an int64 `local_idx` materialize + separate translate launch |
| `cub::BlockRadixSort` (full) | O(N log N) via 8 × 4-bit passes per block | 1 CTA per batch | Sorts everything; top-K is just the first K after sort |
| Hand-coded block radix-select | O(N) | 1 CTA per batch | Stops early once K are found; not in CUB directly |
| Warp-level tournament sort | O(N log K / W) with W warps | 1 CTA per batch | Good for small K; at K=2048 the running heap doesn't fit in registers |

**BOUND choice: `cub::BlockRadixSort<float, 512, 16, int32_t>` with descending sort.**

Justification:
1. **Simplicity of implementation** — one CUB call, correctness is CUB's responsibility.
2. **Eliminates two kernel launches and one int64 allocation** — the primary win is NOT the sort itself beating at::topk; it's the elimination of the `at::topk → local_idx → translate_indices_batched_kernel` chain in favor of a single fused kernel.
3. **8192-element capacity** (512 × 16) covers N_max up to 5760 with headroom. Pad unused slots with `-INFINITY` and index `-1`.
4. **Full sort is only marginally worse than selection for this K/N regime**: K/N ∈ [0.36, 1] across the 128 workloads, so full sort does at most ~2.8× redundant work vs. pure selection. Expected per-batch sort time at 512 threads × 8 radix passes on 8192 items ≈ ~200-300 cycles. Fully amortized against launch-overhead savings.
5. **Fused translate** — immediately after sort, each thread reads its top-K-region value (low ITEMS_PER_THREAD indices correspond to top-ranked items per the BLOCKED output arrangement), looks up `block_table[b, val>>6]`, and writes `phys * 64 + (val & 63)` to `topk_indices[b, :]`. No intermediate tensor.

**REJECTED alternatives:**
- Hand-coded radix-select: higher implementation risk, marginal speed gain over CUB's full sort at this K/N.
- Warp-level tournament sort: K=2048 is too large for in-warp heap; doesn't fit in registers.
- Keep `at::topk`: works, but carries two launches + int64 alloc overhead. `at::topk` also returns `values` which we don't use (wasted bandwidth).

---

## Grid/block topology

- **Grid:** `(B,)` — one CTA per batch. Independent work across batches, no inter-CTA coordination.
- **Block:** 512 threads (`TOPK_BLOCK_THREADS=512`, `TOPK_ITEMS_PER_THREAD=16`).
- **Items_per_thread:** 16. Register budget per thread: 16 keys (f32) + 16 vals (i32) + scratch ≈ 40-50 regs. Under limit.

## SMEM budget

`cub::BlockRadixSort<float, 512, 16, int32_t>::TempStorage` is a union over key/value exchange scratch. For 512×16 items, exchange buffer ≈ 32 KiB (max of the union members). Under 48 KiB → no `cudaFuncSetAttribute` opt-in needed.

## Edge cases

- `seq_len_b == 0`: `actual_topk = 0` → all 128 CTA threads skip output; `topk_indices[b]` remains `-1` from pre-fill. ✓
- `N_max < TOPK_CAPACITY`: pad keys with `-INFINITY`, vals with `-1`. Sort descending places padding at the end. Top-`actual_topk` slots are valid. ✓
- `actual_topk < TOPK`: pre-fill already wrote `-1` to positions `[actual_topk, TOPK)`; kernel writes only ranks `< actual_topk`. ✓
- `N_max > 8192`: IMPOSSIBLE per dataset (max is `max_num_pages=90 → N_max=5760`). Defensive check: `N_max` ≤ `TOPK_CAPACITY` assertion in host code.

## Register allocation for load arrangement

Load STRIPED from GMEM (thread t reads positions `{t, t+512, t+1024, ...}`) → coalesced 128 B transactions. CUB's `SortDescending()` takes BLOCKED input, so we must do an internal `BlockExchange` to go from STRIPED→BLOCKED OR just load BLOCKED and accept non-coalesced GMEM. CUB provides `BlockLoadDirect<BLOCK_LOAD_WARP_TRANSPOSE>` to handle STRIPED→BLOCKED efficiently. **BOUND: use BLOCK_LOAD_DIRECT striped load + CUB's SortDescending that expects BLOCKED** — requires a BlockExchange call between load and sort. Alternative simpler: use `cub::BlockLoad` with `BLOCK_LOAD_WARP_TRANSPOSE` policy which fuses coalesced GMEM read with BLOCKED arrangement.

Actually the cleanest implementation is:
```cpp
using BlockLoad = cub::BlockLoad<float, 512, 16, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
```
which both coalesces GMEM and produces the BLOCKED arrangement SortDescending expects.

## What V3 must verify (Tier 2 has no formal V3; audit items instead)

- Correctness equivalence: block-sorted descending top-`actual_topk` indices match at::topk output up to ordering within ties.
- No register or SMEM over-budget.
- `TOPK_CAPACITY=8192 ≥ max(N_max)=5760` ✓.
- CUB header availability on sm_100a build (CUDA 12.8+).
- Output format: `topk_indices[b, 0..actual_topk-1]` contains translated global token indices, positions `[actual_topk, TOPK)` remain `-1`.

## Projected performance gain

Modest. Current `at::topk + translate_indices` latency ≈ 10-20 µs per launch. Removing both launches + int64 alloc saves ~5-15 µs. Estimated new mean: 0.079 → **0.065-0.070 ms** (~5-15% improvement). Not a home-run; within Tier 2 expectations.
