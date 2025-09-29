file(GLOB MATH_SOURCES "*.cpp" "*.h")

add_library(MATH_LIBRARY ${MATH_SOURCES})

target_include_directories(MATH_LIBRARY
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)