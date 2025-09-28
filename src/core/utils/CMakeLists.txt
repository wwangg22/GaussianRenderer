file(GLOB UTILS_SOURCES "*.cpp" "*.h")

add_library(UTILS_LIBRARY ${UTILS_SOURCES})

target_include_directories(UTILS_LIBRARY
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)