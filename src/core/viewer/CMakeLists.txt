
file(GLOB VIEWER_SOURCES "*.cpp" "*.h")

add_library(VIEWER_LIBRARY ${VIEWER_SOURCES})

target_link_libraries(VIEWER_LIBRARY PRIVATE 
    glfw 
    OpenGL::GL
    glad
)

find_package(OpenGL REQUIRED)


target_include_directories(VIEWER_LIBRARY
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(VIEWER_LIBRARY
  PUBLIC
    SCENE_LIBRARY   # this must be the target name you used in add_library
)
