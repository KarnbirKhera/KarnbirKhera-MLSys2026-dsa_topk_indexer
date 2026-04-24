"""
P6 Production-Replica Probe — Modal Runner

Compiles tmem2_p6_production_replica.cu via load_inline on Modal B200, runs
the production-replica kernel with synthetic Q/K/weights/scale, and verifies
the final_scores against bit-exact expected values.

Test inputs (8xT FP8 layout, 1.0=0x38, 2.0=0x40, 4.0=0x48, 8.0=0x50):
  Q[m, k]    = 1.0 if m == k && m < 64 else 0   → diagonal
  K_p[n, k]  = value_p if n == k && n < 64 else 0
              (page 0: 1.0, page 1: 2.0, page 2: 4.0, page 3: 8.0)
  weights[h] = 1.0  for all h
  token_scale = 1.0

Expected final_scores [256 = 4 pages × 64 cols]:
  [0..63]   = 1.0
  [64..127] = 2.0
  [128..191] = 4.0
  [192..255] = 8.0

If P6 PASSES: TMEM2 reorder + H-reduce + scale apply + scores_smem all work
in a single CTA. The bug requires multi-batch/multi-CTA parallelism not
captured here. Escalate to Option A (instrumented production replica).

If P6 FAILS: minimal reproduction achieved. We can iterate on this small
controllable kernel to localize the bug — one Modal cycle per hypothesis,
directly comparable expected vs actual.
"""

from pathlib import Path
import modal

app = modal.App("tmem2-p6")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12"
    )
    .apt_install("build-essential", "ninja-build", "git")
    .pip_install("torch", "numpy", "ninja")
    .env({"TORCH_CUDA_ARCH_LIST": "10.0a"})
)


def smem_8xT_offset(m: int, k: int) -> int:
    SLAB_BYTES = 2048
    SBO_BYTES = 256
    LBO_BYTES = 128
    return ((k // 32) * SLAB_BYTES
            + (m // 8) * SBO_BYTES
            + ((k % 32) // 16) * LBO_BYTES
            + (m % 8) * 16
            + (k % 16))


def build_diag_tile(byte_value: int) -> bytes:
    """8KB FP8 8xT tile with `byte_value` on diagonal (m,k):m==k && m<64."""
    import numpy as np
    buf = np.zeros(8192, dtype=np.uint8)
    for m in range(64):
        off = smem_8xT_offset(m, m)
        buf[off] = byte_value
    return buf.tobytes()


@app.function(image=image, gpu="B200:1", timeout=600)
def run_p6_remote(kernel_src: str) -> dict:
    import os
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")

    import numpy as np
    import torch
    from torch.utils.cpp_extension import load_inline

    print("[P6] Compiling tmem2_p6_production_replica.cu (sm_100a) ...")
    cpp_decl = """
    #include <torch/extension.h>
    void run_p6(torch::Tensor q_fp8, torch::Tensor k_pages_fp8,
                torch::Tensor weights, double token_scale,
                torch::Tensor final_scores, int64_t B);
    """
    ext = load_inline(
        name="tmem2_p6_ext",
        cpp_sources=cpp_decl,
        cuda_sources=kernel_src,
        functions=["run_p6"],
        verbose=True,
        extra_cuda_cflags=["-arch=sm_100a", "-O3", "--use_fast_math"],
    )
    print("[P6] Compile OK.")

    # Build inputs (shared across all CTAs in P7 multi-CTA mode)
    Q_bytes = build_diag_tile(0x38)
    K_pages_values = [(0, 0x38, 1.0), (1, 0x40, 2.0), (2, 0x48, 4.0), (3, 0x50, 8.0)]
    K_pages_bytes = b"".join(build_diag_tile(b) for _, b, _ in K_pages_values)

    Q       = torch.frombuffer(bytearray(Q_bytes),       dtype=torch.uint8).cuda()
    K_pages = torch.frombuffer(bytearray(K_pages_bytes), dtype=torch.uint8).cuda()
    weights = torch.ones(64, dtype=torch.float32, device="cuda")
    token_scale = 1.0

    results = {}
    for label, B in [("P6_single_CTA", 1), ("P7_multi_CTA_B32", 32),
                     ("P7b_multi_CTA_B128", 128)]:
        final_scores = torch.zeros(B * 256, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        ext.run_p6(Q, K_pages, weights, token_scale, final_scores, B)
        torch.cuda.synchronize()

        expected_one_cta = torch.zeros(256, dtype=torch.float32, device="cuda")
        for p, _, val in K_pages_values:
            expected_one_cta[p*64:(p+1)*64] = val
        expected = expected_one_cta.repeat(B)

        diff = (final_scores - expected).abs()
        max_err = float(diff.max().item())
        ok = max_err < 1e-3
        wrong_total = int((diff > 1e-3).sum().item())

        # Which CTAs are wrong
        per_cta_max = diff.view(B, 256).max(dim=1).values
        wrong_ctas = int((per_cta_max > 1e-3).sum().item())

        # First wrong CTA's per-page breakdown
        first_wrong_cta = -1
        per_page_breakdown = None
        if not ok:
            first_wrong_cta = int(torch.argmax(per_cta_max).item())
            cta_slice = final_scores.view(B, 256)[first_wrong_cta]
            per_page_breakdown = []
            for p, _, val in K_pages_values:
                page_slice = cta_slice[p*64:(p+1)*64]
                page_diff = (page_slice - val).abs()
                wrong = int((page_diff > 1e-3).sum().item())
                page_max = float(page_diff.max().item())
                sample = [float(page_slice[i].item()) for i in range(0, 64, 8)]
                per_page_breakdown.append({
                    "page": p, "expected": val,
                    "wrong_count": wrong, "max_err": page_max,
                    "sample": sample,
                })

        results[label] = {
            "B": B, "ok": ok,
            "max_abs_err": max_err,
            "wrong_total_cells": wrong_total,
            "wrong_ctas": wrong_ctas,
            "first_wrong_cta": first_wrong_cta,
            "first_wrong_breakdown": per_page_breakdown,
        }
        verdict = "PASS" if ok else "FAIL"
        print(f"[{label}] B={B}, {verdict} | max_abs_err = {max_err:.4e} "
              f"| wrong CTAs: {wrong_ctas}/{B} | wrong cells: {wrong_total}/{B*256}")
        if not ok:
            print(f"   First wrong CTA index: {first_wrong_cta}")
            for entry in per_page_breakdown:
                if entry["wrong_count"] > 0:
                    print(f"     page {entry['page']} (expect {entry['expected']}): "
                          f"{entry['wrong_count']}/64 wrong, max_err {entry['max_err']:.3e}")
                    # FULL 64-value dump for the wrong page to find exact wrong positions
                    cta_slice = final_scores.view(B, 256)[first_wrong_cta]
                    full_page = cta_slice[entry['page']*64:(entry['page']+1)*64]
                    print(f"     full page {entry['page']} values:")
                    for chunk_start in range(0, 64, 16):
                        chunk = [f"{full_page[i].item():.2f}" for i in range(chunk_start, chunk_start+16)]
                        print(f"       [{chunk_start:2d}..{chunk_start+15:2d}]: {chunk}")
                    # Wrong positions
                    wrong_idx = (full_page - entry['expected']).abs() > 1e-3
                    wrong_positions = wrong_idx.nonzero(as_tuple=True)[0].tolist()
                    print(f"     wrong positions in page {entry['page']}: {wrong_positions}")
            # ALSO: show which CTAs failed
            wrong_cta_indices = (per_cta_max > 1e-3).nonzero(as_tuple=True)[0].tolist()
            print(f"   All wrong CTA indices: {wrong_cta_indices[:32]}{'...' if len(wrong_cta_indices)>32 else ''}")

    return results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("P6 Production-Replica Probe — Modal B200")
    print("=" * 70)
    print()
    print("Replicates production buggy loop (Stage A reorder + H-reduce + scale +")
    print("scores_smem + GMEM final_scores) on synthetic data with bit-exact")
    print("expected outputs:")
    print("  page 0 → final_scores[0..63]   = 1.0")
    print("  page 1 → final_scores[64..127] = 2.0")
    print("  page 2 → final_scores[128..191]= 4.0")
    print("  page 3 → final_scores[192..255]= 8.0")
    print()
    print("If P6 FAILS → minimal repro of GT-47, ready for stage instrumentation.")
    print("If P6 PASSES → bug requires multi-CTA/multi-batch parallelism.")
    print()

    kernel_src = (Path(__file__).parent / "tmem2_p6_production_replica.cu").read_text()
    result = run_p6_remote.remote(kernel_src)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for label, r in result.items():
        verdict = "PASS" if r["ok"] else "FAIL"
        print(f"  {label:25s} (B={r['B']:4d}): {verdict} | "
              f"max_err {r['max_abs_err']:.3e} | "
              f"wrong CTAs {r['wrong_ctas']}/{r['B']}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    p6_ok = result["P6_single_CTA"]["ok"]
    p7_ok = result["P7_multi_CTA_B32"]["ok"]
    p7b_ok = result["P7b_multi_CTA_B128"]["ok"]

    if p6_ok and not p7_ok:
        print("[!] P6 PASS but P7 FAIL → multi-CTA-induced bug confirmed.")
        print("    The Stage A reorder + production loop body fails specifically when")
        print("    multiple CTAs run in parallel. Single-CTA case is fine.")
        print()
        print("    This is reproducible in a controlled probe — ready for stage")
        print("    instrumentation. Add a debug GMEM buffer to capture intermediate")
        print("    state at each stage; identify which CTA / which page / which stage")
        print("    diverges.")
    elif p6_ok and p7_ok and not p7b_ok:
        print("[!] P7 (B=32) PASS but P7b (B=128) FAIL → bug needs scale.")
        print("    The bug only manifests with very high CTA counts. Likely cross-CTA")
        print("    HBM/L2 contention or scheduling pressure.")
    elif p6_ok and p7_ok and p7b_ok:
        print("[+] All P6/P7/P7b PASS with synthetic data + many CTAs.")
        print("    The bug needs the actual dataset's K data variability or block_table")
        print("    structure. Synthetic uniform data does NOT trigger it.")
        print("    Next: instrument the production kernel directly on a known failing")
        print("    workload (Option A proper).")
    else:
        print("[!] Unexpected pattern; inspect raw results above.")
