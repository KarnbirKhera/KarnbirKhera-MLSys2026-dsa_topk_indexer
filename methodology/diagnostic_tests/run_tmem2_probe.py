"""
TMEM Double-Buffer Diagnostic Probe — Modal Runner

Compiles tmem2_probe.cu via torch.utils.cpp_extension.load on Modal B200,
runs four probe modes (M0/P1/P2/P3) against hand-crafted Q/K with known
expected outputs, and reports per-mode PASS/FAIL.

Test data (constructed in 8xT byte layout matching the kernel):
  Q[m, k]   = 1.0 (FP8 e4m3 byte 0x38) if (m == k && m < 64) else 0.0
  K_A[n, k] = 1.0                       if (n == k && n < 64) else 0.0
  K_B[n, k] = 2.0 (FP8 e4m3 byte 0x40) if (n == k && n < 64) else 0.0

Expected MMA results (D = Q · K^T, FP32 accumulator):
  D_A = I_{64x64}      (1.0 on diagonal)
  D_B = 2 · I_{64x64}  (2.0 on diagonal)

Probe modes:
  M0: single MMA to region A only           — baseline sanity
  P1: single MMA to region B (col + 64)     — region B addressing in isolation
  P2: dispatch A then B (no wait between)   — two MMAs in flight pattern
  P3: P2 with FORCED single lane (lane 0)   — eliminates elect_one_sync re-election
"""

from pathlib import Path

import modal

app = modal.App("tmem2-probe")

# Note: kernel source is read on the LOCAL side inside main() and passed
# to the remote function as an argument. Reading at module-level here would
# also try to read inside the Modal container (where the .cu file isn't
# present) when Modal re-imports the module remotely.

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12"
    )
    .apt_install("build-essential", "ninja-build", "git")
    .pip_install("torch", "numpy", "ninja")
    .env({"TORCH_CUDA_ARCH_LIST": "10.0a"})  # GT-27: required for tcgen05 on sm_100a
)


def smem_8xT_offset(m: int, k: int) -> int:
    """Mirror of the device-side smem_8xT_offset formula."""
    SLAB_BYTES = 2048
    SBO_BYTES = 256
    LBO_BYTES = 128
    return ((k // 32) * SLAB_BYTES
            + (m // 8) * SBO_BYTES
            + ((k % 32) // 16) * LBO_BYTES
            + (m % 8) * 16
            + (k % 16))


def build_diag_tile(byte_value: int) -> bytes:
    """Build an 8KB FP8 tile in 8xT byte layout where (m, k) is byte_value
    iff m == k && m < 64, else 0."""
    import numpy as np
    buf = np.zeros(8192, dtype=np.uint8)
    for m in range(64):
        # Only the diagonal element (m, m) for m in [0, 64) is nonzero.
        # Note: k can be up to 127 but only k == m matters; that's always < 64.
        off = smem_8xT_offset(m, m)
        buf[off] = byte_value
    return buf.tobytes()


@app.function(image=image, gpu="B200:1", timeout=600)
def run_probe_remote(kernel_src: str) -> dict:
    """Compile, run all 4 probe modes, return per-mode results."""
    import os
    import tempfile

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")

    import numpy as np
    import torch
    from torch.utils.cpp_extension import load_inline

    # JIT-compile the kernel via load_inline
    print("[probe] Compiling tmem2_probe.cu (sm_100a) ...")
    # load_inline needs a forward declaration in cpp_sources so the
    # auto-generated main.cpp can resolve run_probe (which is defined in cuda_sources).
    cpp_decl = """
    #include <torch/extension.h>
    void run_probe(torch::Tensor q_fp8, torch::Tensor k_a_fp8, torch::Tensor k_b_fp8,
                   torch::Tensor k_c_fp8,
                   torch::Tensor out_a, torch::Tensor out_b, torch::Tensor out_c,
                   int64_t mode);
    """
    ext = load_inline(
        name="tmem2_probe_ext",
        cpp_sources=cpp_decl,
        cuda_sources=kernel_src,
        functions=["run_probe"],
        verbose=True,
        extra_cuda_cflags=["-arch=sm_100a", "-O3", "--use_fast_math"],
    )
    print("[probe] Compile OK.")

    # Build Q, K_A, K_B, K_C in 8xT layout
    # FP8 e4m3 byte values: 0.0 = 0x00, 1.0 = 0x38, 2.0 = 0x40, 4.0 = 0x48
    Q_bytes   = build_diag_tile(0x38)  # 1.0 on diagonal
    K_A_bytes = build_diag_tile(0x38)  # 1.0 on diagonal
    K_B_bytes = build_diag_tile(0x40)  # 2.0 on diagonal
    K_C_bytes = build_diag_tile(0x48)  # 4.0 on diagonal

    Q   = torch.frombuffer(bytearray(Q_bytes),   dtype=torch.uint8).cuda()
    K_A = torch.frombuffer(bytearray(K_A_bytes), dtype=torch.uint8).cuda()
    K_B = torch.frombuffer(bytearray(K_B_bytes), dtype=torch.uint8).cuda()
    K_C = torch.frombuffer(bytearray(K_C_bytes), dtype=torch.uint8).cuda()

    # Expected outputs
    expected_A = torch.eye(64, dtype=torch.float32).cuda()
    expected_B = 2.0 * torch.eye(64, dtype=torch.float32).cuda()
    expected_C = 4.0 * torch.eye(64, dtype=torch.float32).cuda()

    # (mode_id, name, expect_a_kind, expect_b_kind, expect_c_kind)
    # kind: None = ignored / expect zero, "I" = identity, "2I", "4I"
    PROBE_LIST = [
        (0, "M0_baseline_regionA",   "I",   None, None),
        (1, "P1_regionB_isolation",  None,  "2I", None),
        (2, "P2_two_in_flight",      "I",   "2I", None),
        (3, "P3_forced_single_lane", "I",   "2I", None),
        (4, "P4_mbar_reuse",         "I",   None, "4I"),
        (5, "P5_production_loop",    "I",   "2I", "4I"),
    ]
    EXPECTED = {"I": expected_A, "2I": expected_B, "4I": expected_C}

    results = {}
    for mode_id, mode_name, expect_a, expect_b, expect_c in PROBE_LIST:
        out_a = torch.zeros((64, 64), dtype=torch.float32, device="cuda")
        out_b = torch.zeros((64, 64), dtype=torch.float32, device="cuda")
        out_c = torch.zeros((64, 64), dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        ext.run_probe(Q, K_A, K_B, K_C, out_a, out_b, out_c, mode_id)
        torch.cuda.synchronize()

        entry = {"mode_name": mode_name}

        def check_one(label, actual, expect_kind):
            if expect_kind is None:
                ok = bool(torch.allclose(actual, torch.zeros_like(actual)))
                entry[f"{label}_ok"] = ok
                return ok
            expected = EXPECTED[expect_kind]
            diag_val = float(expected[0, 0].item())  # 1.0, 2.0, or 4.0
            ok = torch.allclose(actual, expected, atol=1e-3)
            entry[f"{label}_ok"] = bool(ok)
            entry[f"{label}_max_abs_err"] = float((actual - expected).abs().max().item())
            entry[f"{label}_diag_sample"] = [float(actual[i, i].item()) for i in range(0, 64, 8)]
            if not ok:
                diff = (actual - expected).abs()
                idx = int(torch.argmax(diff).item())
                m, n = idx // 64, idx % 64
                entry[f"{label}_first_wrong"] = {
                    "m": m, "n": n,
                    "got": float(actual[m, n].item()),
                    "expected": float(expected[m, n].item()),
                }
                diag_wrong = sum(1 for i in range(64) if abs(actual[i, i].item() - diag_val) > 1e-3)
                offdiag_wrong = int(((diff > 1e-3) & (~torch.eye(64, dtype=torch.bool, device="cuda"))).sum().item())
                entry[f"{label}_diag_wrong"] = diag_wrong
                entry[f"{label}_offdiag_wrong"] = offdiag_wrong
            return ok

        a_ok = check_one("regionA", out_a, expect_a)
        b_ok = check_one("regionB", out_b, expect_b)
        c_ok = check_one("regionC", out_c, expect_c)

        results[mode_name] = entry
        verdict = "PASS" if (a_ok and b_ok and c_ok) else "FAIL"
        print(f"[probe] mode {mode_id} ({mode_name}): {verdict}")
        for label, expect_kind in [("regionA", expect_a), ("regionB", expect_b), ("regionC", expect_c)]:
            if expect_kind is not None:
                print(f"        {label} (expect {expect_kind}) diag samples: {entry[f'{label}_diag_sample']}")
                print(f"        {label} max_abs_err = {entry[f'{label}_max_abs_err']:.4e}")
                if not entry[f"{label}_ok"]:
                    print(f"        {label} first wrong: {entry[f'{label}_first_wrong']}")
                    print(f"        {label} diag wrong: {entry[f'{label}_diag_wrong']}/64, "
                          f"offdiag wrong: {entry[f'{label}_offdiag_wrong']}/4032")

    return results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("TMEM Double-Buffer Diagnostic Probe — Modal B200")
    print("=" * 70)
    print()
    print("Test inputs (8xT FP8 layout):")
    print("  Q[m,k]   = 1.0 if m == k (m<64), else 0      → diagonal")
    print("  K_A[n,k] = 1.0 if n == k (n<64), else 0      → diagonal")
    print("  K_B[n,k] = 2.0 if n == k (n<64), else 0      → 2x diagonal")
    print("  K_C[n,k] = 4.0 if n == k (n<64), else 0      → 4x diagonal")
    print()
    print("Expected (FP32):  D_A = I,  D_B = 2I,  D_C = 4I  (each 64x64)")
    print()
    print("Modes:")
    print("  M0: single MMA → region A           (baseline sanity)")
    print("  P1: single MMA → region B           (col + 64 addressing)")
    print("  P2: A then B no-wait, then ld both  (two-in-flight)")
    print("  P3: P2 with forced lane             (elect_one_sync isolation)")
    print("  P4: same mbar reused twice          (mbar reuse + phase flip)")
    print("  P5: production-like 3-MMA flow      (region overwrite + mbar reuse)")
    print()

    kernel_src = (Path(__file__).parent / "tmem2_probe.cu").read_text()
    results = run_probe_remote.remote(kernel_src)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for mode_name, entry in results.items():
        a_ok = entry.get("regionA_ok", True)
        b_ok = entry.get("regionB_ok", True)
        c_ok = entry.get("regionC_ok", True)
        verdict = "PASS" if (a_ok and b_ok and c_ok) else "FAIL"
        if verdict == "FAIL":
            all_pass = False
        print(f"  {mode_name:30s}: {verdict}")
    print()

    # Diagnostic interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    m0 = results.get("M0_baseline_regionA", {})
    p1 = results.get("P1_regionB_isolation", {})
    p2 = results.get("P2_two_in_flight", {})
    p3 = results.get("P3_forced_single_lane", {})
    p4 = results.get("P4_mbar_reuse", {})
    p5 = results.get("P5_production_loop", {})

    def passed(d):
        return d.get("regionA_ok", True) and d.get("regionB_ok", True) and d.get("regionC_ok", True)

    if not passed(m0):
        print("[!] M0 baseline FAILED — kernel/MMA pipeline itself is broken.")
        print("    Suspect: Q/K data construction, IDESC, SMEM descriptor, mbar pattern.")
    elif not passed(p1):
        print("[!] M0 PASS but P1 (region B in isolation) FAILED.")
        print("    Suspect: tmem_col + 64 addressing.")
    elif not passed(p2):
        print("[!] M0 + P1 PASS but P2 (two MMAs in flight) FAILED.")
        if passed(p3):
            print("    P3 (forced single lane) PASSED → elect_one_sync re-election DOES break")
            print("    commit/MMA pairing. Fix: pin a specific lane for all MMA dispatches.")
        else:
            print("    P3 also FAILED → some other two-in-flight bug.")
    elif not passed(p4):
        print("[!] M0/P1/P2/P3 PASS but P4 (mbar reuse) FAILED.")
        print("    The same mbar can NOT be safely reused with phase-parity flip after the first cycle.")
        print("    Possible causes: missing barrier between cycles, phase tracking off-by-one,")
        print("    mbar slot being reset/clobbered.")
    elif not passed(p5):
        print("[!] M0/P1/P2/P3/P4 PASS but P5 (production-loop simulation) FAILED.")
        print("    The bug is specifically in the COMBINED pattern: TMEM region overwrite")
        print("    + mbar reuse + concurrent in-flight MMAs.")
        print("    Likely cause: dispatching a new MMA into a region whose prior value")
        print("    is still being read by another MMA's commit-tracking, or the mbar")
        print("    arrival from the second commit firing before the second MMA actually completes.")
    else:
        print("[+] All probes PASSED.")
        print("    TMEM mechanics + mbar reuse + region overwrite all work in isolation.")
        print("    The GT-47 bug must be in production-kernel-specific state:")
        print("      - K_smem buffer reuse timing across iterations")
        print("      - scale_smem buffer reuse with the H-reduce __syncthreads")
        print("      - Stage E's conditional sync (only fires if has_later)")
        print("      - Interaction with cp.async pipeline depth across iterations")
        print("    Next step: extend probe with cp.async-driven K_smem reuse or")
        print("    bisect the production kernel by progressively enabling stages.")

    if all_pass:
        print()
        print("[+] All probes PASS. Mechanism verified.")
    else:
        print()
        print("[!] Probe failures isolate the GT-47 root cause; see above interpretation.")
