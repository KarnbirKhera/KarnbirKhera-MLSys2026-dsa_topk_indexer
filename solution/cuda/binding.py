"""                                                                                                                                                                                                                                                             
  Torch binding path — the actual C++ entry point is registered in kernel.cu
  via PYBIND11_MODULE. flashinfer-bench's torch build compiles kernel.cu with
  torch.utils.cpp_extension and reads the entry function name from config.toml.                                                                                                                                                                                   
  
  The os.environ.setdefault below is defensive: kernel.cu uses tcgen05.* PTX ops                                                                                                                                                                                  
  (see methodology/CLAUDE.md GT-27), which are rejected on the generic sm_100                                                                                                                                                                                     
  target. Setting TORCH_CUDA_ARCH_LIST=10.0a here ensures sm_100a is selected
  regardless of how the evaluator configures its container.                                                                                                                                                                                                       
  """                                                       
                                                                                                                                                                                                                                                                  
  import os                                                 
  os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")
