file(GLOB CUDA_SOURCES "*.cu" "*.cuh")

add_library(cuda_kernels ${CUDA_SOURCES})

target_compile_options(cuda_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_89>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
target_link_libraries(cuda_kernels
    CUDA::cudart    # CUDA runtime
)