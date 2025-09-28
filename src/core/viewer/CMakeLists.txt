
file(GLOB VIEWER_SOURCES "*.cpp" "*.h")

add_library(VIEWER_LIBRARY ${VIEWER_SOURCES})


include(FetchContent)
FetchContent_Declare(glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.4
)
FetchContent_MakeAvailable(glfw)
target_link_libraries(VIEWER_LIBRARY PRIVATE 
    glfw 
    OpenGL::GL
)

find_package(OpenGL REQUIRED)


target_include_directories(VIEWER_LIBRARY
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)