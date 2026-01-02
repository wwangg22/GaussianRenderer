cmake_minimum_required(VERSION 3.10)
project(renderer)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 89)

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
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/thirdparty/imgui)

add_library(imgui STATIC
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/imgui_demo.cpp

  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
  ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui PUBLIC
  ${IMGUI_DIR}
  ${IMGUI_DIR}/backends
)

target_link_libraries(imgui PUBLIC glfw glad)

include(FetchContent)
FetchContent_Declare(glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.4
)
FetchContent_MakeAvailable(glfw)

add_subdirectory(src)
