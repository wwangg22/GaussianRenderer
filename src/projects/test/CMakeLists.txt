add_executable(radix_test radix_test.cpp)
target_link_libraries(radix_test cuda_kernels)

add_executable(onesweep onesweep.cpp)
target_link_libraries(onesweep cuda_kernels)

add_executable(window_test window_test.cpp)
target_link_libraries(window_test VIEWER_LIBRARY)

add_executable(parser_test parser_test.cpp)
target_link_libraries(parser_test UTILS_LIBRARY)