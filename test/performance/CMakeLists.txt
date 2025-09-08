# Performance monitoring tests
cmake_minimum_required(VERSION 3.14)

# Find required packages
find_package(GTest REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLOG REQUIRED libglog)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/performance
    ${CMAKE_SOURCE_DIR}/include/util
    ${CMAKE_SOURCE_DIR}/include/jit
    ${CMAKE_SOURCE_DIR}/include/base
    ${GLOG_INCLUDE_DIRS}
)

# Performance monitor test executable
add_executable(test_performance_monitor test_performance_monitor.cpp)

# Link libraries
target_link_libraries(test_performance_monitor
    PRIVATE
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
    CUDA::nvrtc
    ${GLOG_LIBRARIES}
    pthread
    dl
    rt
    cuop_performance
)

# Set properties
set_target_properties(test_performance_monitor PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add to CTest
enable_testing()
add_test(NAME PerformanceMonitorTest COMMAND test_performance_monitor)
