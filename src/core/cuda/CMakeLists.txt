file(GLOB CUDA_SOURCES "*.cu" "*.cuh")

add_library(cuda_kernels ${CUDA_SOURCES})

target_compile_options(cuda_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_89>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
target_include_directories(cuda_kernels
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(cuda_kernels
  PUBLIC
    CUDA::cudart    # CUDA runtime
    UTILS_LIBRARY
    SCENE_LIBRARY
)