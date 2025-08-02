cmake_minimum_required(VERSION 3.10)
project(test_globalaverpool2D)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/../../../include
    ${PROJECT_SOURCE_DIR}/../../../third_party
)

add_executable(test_globalaverpool2D test_globalaverpool2D.cpp)
target_link_libraries(test_globalaverpool2D
    cuda
    cudart
    gtest
    gtest_main
    pthread
    glog
) 