"""
Binding shim for the DSA TopK FP8 indexer kernel.

Binding mode is "torch" (see config.toml [build] binding). The torch binding
path in flashinfer-bench drives kernel compilation via
torch.utils.cpp_extension.load(), which selects the target SM arch from the
TORCH_CUDA_ARCH_LIST env var. Unset, it auto-detects and picks sm_100 on B200,
which ptxas rejects for every tcgen05.* op in kernel.cu (see CLAUDE.md GT-27).
Setting the env var here ensures sm_100a is targeted regardless of how the
evaluator configures its container.
"""

import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")


@register_func("flashinfer.kernel")
def kernel():
    """
    Python binding for your CUDA kernel.

    TODO: Implement the binding according to the track definition.
    This function should:
    1. Accept the inputs as specified by the track definition
    2. Launch your CUDA kernel with appropriate grid/block dimensions
    3. Return outputs as specified by the track definition
    """
    pass
