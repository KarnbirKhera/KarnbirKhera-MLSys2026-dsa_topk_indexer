"""
NCU Profile of the Production Baseline Kernel — Modal B200

Goal: identify cost hotspots before committing to Cand 3 (Streaming Gate via
cluster+DSM). Specifically answer:
  1. Where does kernel time actually go? (scoring vs top-K vs launch overhead)
  2. Is final_scores GMEM round trip a significant fraction? (Cand 3's premise)
  3. Is tensor pipe the bottleneck? Or HBM? Or warp scheduler stalls?
  4. Are there cheaper opts hiding in the metrics that we'd miss otherwise?

Approach: use the same flashinfer_bench harness as run_modal.py but wrap a few
kernel launches in `ncu` to capture metrics. Profile ~5 launches (enough for
representative numbers, not so many that NCU overhead extends Modal timeout).

Output: parsed CSV from ncu, summarized by metric category, plus the raw output
saved for forensic inspection.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent / "flashinfer-bench-starter-kit"
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("ncu-profile")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

MOUNT_PATH = "/data"
TRACE_SET_PATH = "/data/mlsys26-contest"

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
    .env({"TORCH_CUDA_ARCH_LIST": "10.0a"})
)

# NOTE: kernel source is read inside main() (local entrypoint) and passed
# to the remote function as an argument — Modal re-imports the module
# remotely where the local file isn't present, so module-level reads fail.


@app.function(
    image=image,
    gpu="B200:1",
    timeout=1800,  # 30 min — NCU is slow
    volumes={MOUNT_PATH: trace_volume},
)
def profile_with_ncu(kernel_src: str, config_toml: str) -> dict:
    """Run a small benchmark under ncu and capture per-kernel metrics."""
    import os
    import subprocess
    import tempfile
    import textwrap

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")

    # Driver: load_inline-compile the kernel, load ONE large workload's
    # safetensors directly from the volume, run a few launches under ncu.
    driver = textwrap.dedent('''
        import os
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")

        import torch
        from pathlib import Path
        from torch.utils.cpp_extension import load_inline
        import safetensors.torch as st

        BLOB_DIR = "/data/mlsys26-contest/blob/workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64"

        # Compile the kernel (PYBIND11_MODULE stripped on host side already).
        kernel_src = Path("/tmp/kernel.cu").read_text()
        cpp_decl = """
        #include <torch/extension.h>
        void launch_topk_c(torch::Tensor q_index_fp8,
                           torch::Tensor k_index_cache_fp8,
                           torch::Tensor weights,
                           torch::Tensor seq_lens,
                           torch::Tensor block_table,
                           torch::Tensor topk_indices);
        """
        print("[driver] Compiling kernel ...")
        ext = load_inline(
            name="ncu_profile_ext",
            cpp_sources=cpp_decl,
            cuda_sources=kernel_src,
            functions=["launch_topk_c"],
            verbose=False,
            extra_cuda_cflags=["-arch=sm_100a", "-O3", "--use_fast_math"],
        )
        print("[driver] Compile OK.")

        # Pick a LARGE workload — biggest safetensors file = most pages/batches
        files = sorted(os.listdir(BLOB_DIR))
        files_sized = [(f, os.path.getsize(os.path.join(BLOB_DIR, f))) for f in files]
        files_sized.sort(key=lambda x: -x[1])
        chosen_file = files_sized[0][0]
        print(f"[driver] Selected workload: {chosen_file} ({files_sized[0][1]} bytes)")

        inputs = st.load_file(os.path.join(BLOB_DIR, chosen_file))
        print(f"[driver] Workload structural tensors: {list(inputs.keys())}")
        for k, v in inputs.items():
            print(f"  {k}: shape={list(v.shape)} dtype={v.dtype}")

        # Workload safetensors have only block_table + seq_lens (structural).
        # Synthesize q/K/weights — for NCU profiling we don't need correct outputs,
        # just realistic kernel timing on representative shapes.
        block_table = inputs["block_table"].cuda()
        seq_lens = inputs["seq_lens"].cuda()
        B = block_table.shape[0]
        max_num_pages = block_table.shape[1]
        num_pages = int(block_table.max().item()) + 1

        # FP8 byte tensors viewed as int8 (kernel reinterprets as uint8 internally).
        # Use simple deterministic data so cp.async hits cache predictably.
        q_index_fp8 = torch.randint(0, 64, (B, 64, 128), dtype=torch.int8, device="cuda")
        k_index_cache_fp8 = torch.randint(0, 64, (num_pages, 64, 1, 132), dtype=torch.int8, device="cuda")
        weights = torch.ones((B, 64), dtype=torch.float32, device="cuda") * 0.5
        topk_indices = torch.full((B, 2048), -1, dtype=torch.int32, device="cuda")

        print(f"[driver] B={B}, max_num_pages={max_num_pages}, num_pages={num_pages}")
        print(f"[driver] K-cache size: {k_index_cache_fp8.numel() / 1024:.1f} KB")

        # Warmup
        for _ in range(3):
            ext.launch_topk_c(
                q_index_fp8, k_index_cache_fp8, weights,
                seq_lens, block_table, topk_indices,
            )
        torch.cuda.synchronize()

        # Profiled section — ncu captures only launches between Start/Stop
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(3):
            ext.launch_topk_c(
                q_index_fp8, k_index_cache_fp8, weights,
                seq_lens, block_table, topk_indices,
            )
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

        print("[driver] Done.")
    ''')

    # Save kernel.cu and driver
    Path("/tmp/kernel.cu").write_text(kernel_src)
    Path("/tmp/driver.py").write_text(driver)

    # Curated NCU metric set focused on the questions we need to answer
    metrics = ",".join([
        # Time / activity
        "gpu__time_duration.sum",
        "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
        # Tensor pipe (MMA cost)
        "sm__inst_executed_pipe_tensor.sum",
        "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed",
        # HBM bandwidth (load/store cost on K-cache, final_scores)
        "dram__bytes.sum",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        # L2 traffic (final_scores round trip lives here for small B)
        "lts__t_sectors.sum",
        "lts__t_bytes_lookup_hit.sum",
        # SM scheduler — why did warps stall?
        "smsp__average_warps_issue_stalled_barrier.ratio",
        "smsp__average_warps_issue_stalled_membar.ratio",
        "smsp__average_warps_issue_stalled_short_scoreboard.ratio",
        "smsp__average_warps_issue_stalled_long_scoreboard.ratio",
        "smsp__average_warps_issue_stalled_dispatch_stall.ratio",
        # SMEM bank conflicts
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
        # Achieved occupancy
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ])

    cmd = [
        "ncu",
        "--csv",
        "--page", "raw",
        "--target-processes", "all",
        "--metrics", metrics,
        "--profile-from-start", "off",  # only profile cudaProfilerStart..Stop region
        "python", "/tmp/driver.py",
    ]

    print("[ncu] Running:", " ".join(cmd[:5]), "... [metrics list omitted]")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("NCU Profile — Production Baseline Kernel — Modal B200")
    print("=" * 70)
    print()

    solution_dir = Path(__file__).parent.parent / "flashinfer-bench-starter-kit" / "solution"
    kernel_src = (solution_dir / "cuda" / "kernel.cu").read_text()
    # Strip the PYBIND11_MODULE block — load_inline auto-generates its own,
    # otherwise we get duplicate PyInit symbol at link time.
    if "PYBIND11_MODULE" in kernel_src:
        kernel_src = kernel_src.partition("PYBIND11_MODULE")[0]
        kernel_src += "\n// PYBIND11_MODULE stripped for load_inline\n"
    config_toml = (Path(__file__).parent.parent / "flashinfer-bench-starter-kit" / "config.toml").read_text()
    result = profile_with_ncu.remote(kernel_src, config_toml)

    print(f"[result] returncode = {result['returncode']}")
    print()
    print("=" * 70)
    print("STDOUT (NCU CSV output)")
    print("=" * 70)
    # Save full output for forensic
    Path("/tmp/ncu_stdout.txt").write_text(result["stdout"])
    Path("/tmp/ncu_stderr.txt").write_text(result["stderr"])
    print(f"[saved] /tmp/ncu_stdout.txt ({len(result['stdout'])} chars)")
    print(f"[saved] /tmp/ncu_stderr.txt ({len(result['stderr'])} chars)")

    # Print first 200 lines of stdout for quick look
    lines = result["stdout"].splitlines()
    print(f"[stdout preview, first 200 of {len(lines)} lines]")
    for line in lines[:200]:
        print(line)

    print()
    print("=" * 70)
    print("STDERR (compile + driver logs)")
    print("=" * 70)
    err_lines = result["stderr"].splitlines()
    print(f"[stderr preview, last 60 of {len(err_lines)} lines]")
    for line in err_lines[-60:]:
        print(line)
