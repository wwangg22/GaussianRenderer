cmake_minimum_required(VERSION 3.10)
project(renderer)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
# libraries loading below
# ---------- GLAD (vendored) ----------
# Expecting:
#   thirdparty/glad/include/glad/glad.h
#   thirdparty/glad/include/KHR/khrplatform.h
#   thirdparty/glad/src/glad.c
add_library(glad STATIC
  ${CMAKE_SOURCE_DIR}/thirdparty/glad/src/glad.c
)
target_include_directories(glad PUBLIC
  ${CMAKE_SOURCE_DIR}/thirdparty/glad/include
)

include(FetchContent)
FetchContent_Declare(glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.4
)
FetchContent_MakeAvailable(glfw)

add_subdirectory(src)
