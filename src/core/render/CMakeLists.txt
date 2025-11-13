file(GLOB RENDER_SOURCES "*.cpp" "*.hpp")

add_library(RENDER_LIBRARY ${RENDER_SOURCES})

target_link_libraries(RENDER_LIBRARY 
    PUBLIC 
    glfw 
    OpenGL::GL
    glad
)

find_package(OpenGL REQUIRED)

target_include_directories(RENDER_LIBRARY
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(RENDER_LIBRARY
  PUBLIC
    UTILS_LIBRARY
    SCENE_LIBRARY
    MATH_LIBRARY
)