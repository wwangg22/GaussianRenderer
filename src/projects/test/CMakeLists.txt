add_executable(radix_test radix_test.cpp)
target_link_libraries(radix_test cuda_kernels)

add_executable(onesweep onesweep.cpp)
target_link_libraries(onesweep cuda_kernels)

add_executable(window_test window_test.cpp)
target_link_libraries(window_test VIEWER_LIBRARY)

add_executable(parser_test parser_test.cpp)
target_link_libraries(parser_test UTILS_LIBRARY)

add_executable(camera_test camera_test.cpp)
target_link_libraries(camera_test SCENE_LIBRARY)

add_executable(cull_sort_test cull_sort_test.cpp)
target_link_libraries(cull_sort_test RENDER_LIBRARY cuda_kernels)