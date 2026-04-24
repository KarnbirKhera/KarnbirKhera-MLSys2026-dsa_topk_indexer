"""
P11 — K_smem Overwrite Race Probe — Modal Runner

Tests Hypothesis #6: tcgen05.commit's mbarrier arrival fires before MMA's
SOURCE-SMEM reads complete. If so, overwriting K_smem AFTER mbar.wait will
cause MMA's still-in-flight tensor-core reads to pick up the new K bytes,
producing output computed with the WRONG K.

Test sequence (per CTA):
  1. Load K_A (1.0 diag) into K_smem
  2. Dispatch MMA → region 0 → commit mbar
  3. mbar.wait
  4. (mode=1 only) overwrite K_smem with K_C bytes (4.0 diag)
  5. tcgen05.ld region 0 → write to GMEM
  6. Verify diagonal:
       1.0 = MMA used K_A (correct, mbar was honest)
       4.0 = MMA used K_C (mbar fired before K_smem reads finished — BUG!)

Three test variants:
  P11a (mode=0, B=1):    control. expect 1.0 everywhere.
  P11b (mode=1, B=1):    single-CTA overwrite. expect 1.0 if mbar correct.
                         If 4.0, single-CTA already has the race.
  P11c (mode=1, B=32):   multi-CTA overwrite stress. If 4.0 anywhere, mbar
                         fires early under multi-CTA contention → hypothesis
                         confirmed. Predict same 16-position pattern as GT-47:
                         warps 2&3 (rows 32-63), HI half (cols 32-63), specific
                         lane offsets.
"""

from pathlib import Path
import modal

app = modal.App("tmem2-p11")

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
    import numpy as np
    buf = np.zeros(8192, dtype=np.uint8)
    for m in range(64):
        off = smem_8xT_offset(m, m)
        buf[off] = byte_value
    return buf.tobytes()


@app.function(image=image, gpu="B200:1", timeout=600)
def run_p11_remote(kernel_src: str) -> dict:
    import os
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")

    import numpy as np
    import torch
    from torch.utils.cpp_extension import load_inline

    print("[P11] Compiling tmem2_p11_kmem_overwrite_race.cu (sm_100a) ...")
    cpp_decl = """
    #include <torch/extension.h>
    void run_p11(torch::Tensor q_fp8, torch::Tensor k_a_fp8, torch::Tensor k_c_fp8,
                 int64_t mode, torch::Tensor out, int64_t B);
    """
    ext = load_inline(
        name="tmem2_p11_ext",
        cpp_sources=cpp_decl,
        cuda_sources=kernel_src,
        functions=["run_p11"],
        verbose=True,
        extra_cuda_cflags=["-arch=sm_100a", "-O3", "--use_fast_math"],
    )
    print("[P11] Compile OK.")

    # Inputs: Q (1.0 diag), K_A (1.0 diag), K_C (4.0 diag)
    Q   = torch.frombuffer(bytearray(build_diag_tile(0x38)), dtype=torch.uint8).cuda()
    K_A = torch.frombuffer(bytearray(build_diag_tile(0x38)), dtype=torch.uint8).cuda()
    K_C = torch.frombuffer(bytearray(build_diag_tile(0x48)), dtype=torch.uint8).cuda()

    # Expected MMA output: D[m, n] = K_diag_value if m==n else 0
    expected_KA = torch.zeros(64, 64, dtype=torch.float32, device="cuda")
    for i in range(64):
        expected_KA[i, i] = 1.0

    results = {}
    for label, mode, B, expect_value in [
        ("P11a_control_no_overwrite_B1",      0, 1,   1.0),
        ("P11b_overwrite_B1",                 1, 1,   1.0),
        ("P11c_overwrite_B32",                1, 32,  1.0),
        ("P11d_overwrite_B128",               1, 128, 1.0),
        # Step 1 (Cand 1+2 viability): does the drain matter when there's NO K_smem
        # overwrite at all? mode=0 = no overwrite + no syncthreads after wait_ld.
        # If P11e/f PASS → drain is purely for K_smem race; the proposed 4-K_smem
        # rework can eliminate the drain. If they FAIL → drain is needed regardless
        # of K_smem reuse (e.g., for general tensor-pipe quiescence), rework is moot.
        ("P11e_no_overwrite_B32",             0, 32,  1.0),
        ("P11f_no_overwrite_B128",            0, 128, 1.0),
    ]:
        out = torch.zeros(B * 64 * 64, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        ext.run_p11(Q, K_A, K_C, mode, out, B)
        torch.cuda.synchronize()

        out_view = out.view(B, 64, 64)

        # Build expected per-CTA (broadcast)
        expected_per_cta = torch.zeros(64, 64, dtype=torch.float32, device="cuda")
        for i in range(64):
            expected_per_cta[i, i] = expect_value
        expected = expected_per_cta.unsqueeze(0).expand(B, -1, -1)

        diff = (out_view - expected).abs()
        max_err = float(diff.max().item())
        ok = max_err < 1e-3
        wrong_total = int((diff > 1e-3).sum().item())
        per_cta_max = diff.view(B, -1).max(dim=1).values
        wrong_ctas = int((per_cta_max > 1e-3).sum().item())

        # Distinguish: contamination from K_C (output = 4.0 on diag)
        # vs other corruption (zeros, junk, etc.)
        kc_contaminated_count = 0
        zero_count = 0
        if not ok:
            first_wrong_cta = int(torch.argmax(per_cta_max).item())
            cta_out = out_view[first_wrong_cta]
            # Look at diagonal values
            diag_vals = torch.diag(cta_out).tolist()
            # Wrong positions on diagonal
            wrong_diag_positions = [i for i in range(64) if abs(diag_vals[i] - 1.0) > 1e-3]
            wrong_diag_vals = [diag_vals[i] for i in wrong_diag_positions]
            kc_contaminated_count = sum(1 for v in wrong_diag_vals if abs(v - 4.0) < 1e-3)
            zero_count = sum(1 for v in wrong_diag_vals if abs(v) < 1e-3)
            print(f"[{label}] B={B}, mode={mode}, FAIL")
            print(f"   max_err={max_err:.3e}, wrong CTAs={wrong_ctas}/{B}, wrong cells={wrong_total}/{B*64*64}")
            print(f"   first wrong CTA index: {first_wrong_cta}")
            print(f"   wrong diagonal positions: {wrong_diag_positions}")
            print(f"   wrong diagonal values: {[f'{v:.2f}' for v in wrong_diag_vals]}")
            print(f"   = K_C contamination (4.0): {kc_contaminated_count}, "
                  f"zeros: {zero_count}, other: {len(wrong_diag_vals) - kc_contaminated_count - zero_count}")
        else:
            print(f"[{label}] B={B}, mode={mode}, PASS (max_err={max_err:.3e})")

        results[label] = {
            "mode": mode, "B": B, "ok": ok,
            "max_err": max_err,
            "wrong_ctas": wrong_ctas,
            "wrong_total": wrong_total,
            "kc_contaminated": kc_contaminated_count,
            "zero_corrupt": zero_count,
        }

    return results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("P11 — K_smem Overwrite Race Probe — Modal B200")
    print("=" * 70)
    print()
    print("Tests Hypothesis #6: mbar fires before MMA's K_smem reads complete")
    print()

    kernel_src = (Path(__file__).parent / "tmem2_p11_kmem_overwrite_race.cu").read_text()
    results = run_p11_remote.remote(kernel_src)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for label, r in results.items():
        verdict = "PASS" if r["ok"] else "FAIL"
        suffix = ""
        if not r["ok"]:
            suffix = (f"  [K_C contamination: {r['kc_contaminated']}, "
                      f"zeros: {r['zero_corrupt']}]")
        print(f"  {label:35s}: {verdict} | wrong CTAs {r['wrong_ctas']}/{r['B']}{suffix}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    p11a = results["P11a_control_no_overwrite_B1"]
    p11b = results["P11b_overwrite_B1"]
    p11c = results["P11c_overwrite_B32"]
    p11d = results["P11d_overwrite_B128"]

    if not p11a["ok"]:
        print("[!] P11a control FAILED — kernel/MMA setup itself is broken. Inspect.")
    elif not p11b["ok"]:
        print("[!] P11a PASS but P11b FAIL — even single-CTA, mbar fires before K_smem reads complete.")
        print("    This DIRECTLY confirms Hypothesis #6.")
        print(f"    Single-CTA contamination: {p11b['kc_contaminated']} K_C-poisoned, "
              f"{p11b['zero_corrupt']} zero corrupt.")
    elif p11b["ok"] and (not p11c["ok"] or not p11d["ok"]):
        print("[!] P11b PASS (single-CTA mbar correctly tracks completion) but P11c/d FAIL.")
        print("    Multi-CTA stress causes mbar to fire early — exactly the GT-47 pattern.")
        print("    Hypothesis #6 CONFIRMED under multi-CTA conditions.")
        worst = p11d if not p11d["ok"] else p11c
        if worst["kc_contaminated"] > 0:
            print(f"    Direct evidence: {worst['kc_contaminated']} diagonal positions returned")
            print(f"    K_C value (4.0) instead of K_A's (1.0). MMA's tensor-core SMEM reads")
            print(f"    physically pulled in the post-overwrite K_C bytes — same root-cause")
            print(f"    pattern as the production buggy kernel.")
        print()
        print("    PRACTICAL FIX CANDIDATES:")
        print("    - Add explicit __threadfence_block() between mbar.wait and K_smem overwrite")
        print("    - Add a second mbarrier handshake to confirm 'MMA fully drained' before write")
        print("    - Add a delay loop to give tensor pipe time to drain")
    else:
        print("[+] All P11 modes PASS. Hypothesis #6 NOT confirmed — mbar correctly tracks")
        print("    MMA SMEM-read completion even under multi-CTA stress.")
        print("    The GT-47 bug must be in a different mechanism (likely scoreboard or")
        print("    TMEM write delivery, not source-SMEM reads).")
