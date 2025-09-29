file(GLOB SCENE_SOURCES "*.cpp" "*.h")

add_library(SCENE_LIBRARY ${SCENE_SOURCES})

target_include_directories(SCENE_LIBRARY
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(UTILS_LIBRARY
  PUBLIC
    MATH_LIBRARY   # this must be the target name you used in add_library
)