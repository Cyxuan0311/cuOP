cmake_minimum_required(VERSION 3.10)
project(test_relu)

find_package(CUDA REQUIRED)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
    ${glog_INCLUDE_DIRS}
    ../../../../include
    ../../../../
)

link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

add_executable(test_relu test_relu.cpp)

target_link_libraries(test_relu
    ${CUDA_LIBRARIES}
    ${GTEST_BOTH_LIBRARIES}
    glog::glog
    cublas
    cudart
)

enable_testing()
add_test(NAME test_relu COMMAND test_relu)
