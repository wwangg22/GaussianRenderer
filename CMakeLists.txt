cmake_minimum_required(VERSION 3.10)
project(TestProject)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(test_app test.cpp)
