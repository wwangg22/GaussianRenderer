file(GLOB CUDA_SOURCES "*.cu" "*.cuh")

add_library(cuda_kernels ${CUDA_SOURCES})

target_link_libraries(cuda_kernels
    CUDA::cudart    # CUDA runtime
)