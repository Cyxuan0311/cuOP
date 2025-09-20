#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include "cuda_op/detail/cuBlas/gemm.hpp"
#include "cuda_op/detail/cuBlas/gemv.hpp"
#include "cuda_op/detail/cuBlas/dot.hpp"
#include "cuda_op/detail/cuBlas/axpy.hpp"
#include "cuda_op/detail/cuDNN/relu.hpp"
#include "cuda_op/detail/cuDNN/softmax.hpp"
#include "data/tensor.hpp"
#include "util/status_code.hpp"

using namespace cu_op_mem;

struct BenchmarkResult {
    std::string operation;
    std::string size;
    double time_ms;
    double gflops;
    double bandwidth_gb_s;
    bool success;
};

class PerformanceBenchmark {
private:
    std::vector<BenchmarkResult> results_;
    
public:
    void RunGemmBenchmark() {
        std::cout << "\n=== GEMM 性能基准测试 ===" << std::endl;
        
        std::vector<int> sizes = {256, 512, 1024, 2048};
        
        for (int size : sizes) {
            try {
                // 创建矩阵
                Tensor<float> A({static_cast<size_t>(size), static_cast<size_t>(size)});
                Tensor<float> B({static_cast<size_t>(size), static_cast<size_t>(size)});
                Tensor<float> C({static_cast<size_t>(size), static_cast<size_t>(size)});
                
                // 初始化数据
                std::vector<float> h_A(size * size, 1.0f);
                std::vector<float> h_B(size * size, 2.0f);
                std::vector<float> h_C(size * size, 0.0f);
                
                cudaMemcpy(A.data(), h_A.data(), A.bytes(), cudaMemcpyHostToDevice);
                cudaMemcpy(B.data(), h_B.data(), B.bytes(), cudaMemcpyHostToDevice);
                cudaMemcpy(C.data(), h_C.data(), C.bytes(), cudaMemcpyHostToDevice);
                
                // 创建GEMM算子
                Gemm<float> gemm(false, false, 1.0f, 0.0f);
                gemm.SetWeight(B);
                
                // 预热
                for (int i = 0; i < 3; ++i) {
                    gemm.Forward(A, C);
                }
                cudaDeviceSynchronize();
                
                // 性能测试
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < 10; ++i) {
                    gemm.Forward(A, C);
                }
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                double time_ms = duration.count() / 1000.0 / 10.0; // 平均每次执行时间
                
                // 计算GFLOPS
                long long flops = 2LL * size * size * size;
                double gflops = flops / (time_ms * 1e6);
                
                // 计算带宽
                long long bytes = (size * size * 3) * sizeof(float); // A, B, C
                double bandwidth_gb_s = (bytes / (time_ms * 1e-3)) / 1e9;
                
                std::cout << std::setw(8) << size << "x" << size 
                         << std::setw(12) << std::fixed << std::setprecision(2) << time_ms << " ms"
                         << std::setw(12) << std::fixed << std::setprecision(2) << gflops << " GFLOPS"
                         << std::setw(12) << std::fixed << std::setprecision(2) << bandwidth_gb_s << " GB/s"
                         << std::endl;
                
                results_.push_back({"GEMM", std::to_string(size) + "x" + std::to_string(size), 
                                  time_ms, gflops, bandwidth_gb_s, true});
                
            } catch (const std::exception& e) {
                std::cout << "GEMM " << size << "x" << size << " 失败: " << e.what() << std::endl;
                results_.push_back({"GEMM", std::to_string(size) + "x" + std::to_string(size), 
                                  0, 0, 0, false});
            }
        }
    }
    
    void RunGemvBenchmark() {
        std::cout << "\n=== GEMV 性能基准测试 ===" << std::endl;
        
        std::vector<std::pair<int, int>> sizes = {{1024, 1024}, {2048, 2048}, {4096, 4096}, {8192, 8192}};
        
        for (auto [M, N] : sizes) {
            try {
                // 创建矩阵和向量
                Tensor<float> A({static_cast<size_t>(M), static_cast<size_t>(N)});
                Tensor<float> x({static_cast<size_t>(N)});
                Tensor<float> y({static_cast<size_t>(M)});
                
                // 初始化数据
                std::vector<float> h_A(M * N, 1.0f);
                std::vector<float> h_x(N, 2.0f);
                std::vector<float> h_y(M, 0.0f);
                
                cudaMemcpy(A.data(), h_A.data(), A.bytes(), cudaMemcpyHostToDevice);
                cudaMemcpy(x.data(), h_x.data(), x.bytes(), cudaMemcpyHostToDevice);
                cudaMemcpy(y.data(), h_y.data(), y.bytes(), cudaMemcpyHostToDevice);
                
                // 创建GEMV算子
                Gemv<float> gemv(false, 1.0f, 0.0f);
                gemv.SetWeight(x);
                
                // 预热
                for (int i = 0; i < 3; ++i) {
                    gemv.Forward(A, y);
                }
                cudaDeviceSynchronize();
                
                // 性能测试
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < 100; ++i) {
                    gemv.Forward(A, y);
                }
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                double time_ms = duration.count() / 1000.0 / 100.0; // 平均每次执行时间
                
                // 计算GFLOPS
                long long flops = 2LL * M * N;
                double gflops = flops / (time_ms * 1e6);
                
                // 计算带宽
                long long bytes = (M * N + M + N) * sizeof(float);
                double bandwidth_gb_s = (bytes / (time_ms * 1e-3)) / 1e9;
                
                std::cout << std::setw(8) << M << "x" << N 
                         << std::setw(12) << std::fixed << std::setprecision(2) << time_ms << " ms"
                         << std::setw(12) << std::fixed << std::setprecision(2) << gflops << " GFLOPS"
                         << std::setw(12) << std::fixed << std::setprecision(2) << bandwidth_gb_s << " GB/s"
                         << std::endl;
                
                results_.push_back({"GEMV", std::to_string(M) + "x" + std::to_string(N), 
                                  time_ms, gflops, bandwidth_gb_s, true});
                
            } catch (const std::exception& e) {
                std::cout << "GEMV " << M << "x" << N << " 失败: " << e.what() << std::endl;
                results_.push_back({"GEMV", std::to_string(M) + "x" + std::to_string(N), 
                                  0, 0, 0, false});
            }
        }
    }
    
    void RunDotBenchmark() {
        std::cout << "\n=== DOT 性能基准测试 ===" << std::endl;
        
        std::vector<int> sizes = {1024, 4096, 16384, 65536, 262144};
        
        for (int size : sizes) {
            try {
                // 创建向量
                Tensor<float> x({static_cast<size_t>(size)});
                Tensor<float> y({static_cast<size_t>(size)});
                
                // 初始化数据
                std::vector<float> h_x(size, 1.0f);
                std::vector<float> h_y(size, 2.0f);
                
                cudaMemcpy(x.data(), h_x.data(), x.bytes(), cudaMemcpyHostToDevice);
                cudaMemcpy(y.data(), h_y.data(), y.bytes(), cudaMemcpyHostToDevice);
                
                // 创建DOT算子
                Dot<float> dot;
                
                // 预热
                float result;
                for (int i = 0; i < 3; ++i) {
                    dot.Forward(x, y, result);
                }
                cudaDeviceSynchronize();
                
                // 性能测试
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < 1000; ++i) {
                    dot.Forward(x, y, result);
                }
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                double time_ms = duration.count() / 1000.0 / 1000.0; // 平均每次执行时间
                
                // 计算GFLOPS
                long long flops = size;
                double gflops = flops / (time_ms * 1e6);
                
                // 计算带宽
                long long bytes = 2 * size * sizeof(float);
                double bandwidth_gb_s = (bytes / (time_ms * 1e-3)) / 1e9;
                
                std::cout << std::setw(8) << size 
                         << std::setw(12) << std::fixed << std::setprecision(2) << time_ms << " ms"
                         << std::setw(12) << std::fixed << std::setprecision(2) << gflops << " GFLOPS"
                         << std::setw(12) << std::fixed << std::setprecision(2) << bandwidth_gb_s << " GB/s"
                         << std::endl;
                
                results_.push_back({"DOT", std::to_string(size), 
                                  time_ms, gflops, bandwidth_gb_s, true});
                
            } catch (const std::exception& e) {
                std::cout << "DOT " << size << " 失败: " << e.what() << std::endl;
                results_.push_back({"DOT", std::to_string(size), 
                                  0, 0, 0, false});
            }
        }
    }
    
    void RunReluBenchmark() {
        std::cout << "\n=== ReLU 性能基准测试 ===" << std::endl;
        
        std::vector<std::vector<int>> shapes = {
            {32, 64, 224, 224},
            {16, 128, 112, 112},
            {8, 256, 56, 56},
            {4, 512, 28, 28}
        };
        
        for (const auto& shape : shapes) {
            try {
                // 转换shape为size_t
                std::vector<size_t> shape_size_t(shape.begin(), shape.end());
                
                // 创建张量
                Tensor<float> input(shape_size_t);
                Tensor<float> output(shape_size_t);
                
                // 初始化数据
                int total_elements = 1;
                for (int dim : shape) total_elements *= dim;
                
                std::vector<float> h_input(total_elements);
                for (int i = 0; i < total_elements; ++i) {
                    h_input[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 10.0f;
                }
                
                cudaMemcpy(input.data(), h_input.data(), input.bytes(), cudaMemcpyHostToDevice);
                
                // 创建ReLU算子
                Relu<float> relu;
                
                // 预热
                for (int i = 0; i < 3; ++i) {
                    relu.Forward(input, output);
                }
                cudaDeviceSynchronize();
                
                // 性能测试
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < 100; ++i) {
                    relu.Forward(input, output);
                }
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                double time_ms = duration.count() / 1000.0 / 100.0; // 平均每次执行时间
                
                // 计算GFLOPS
                long long flops = total_elements;
                double gflops = flops / (time_ms * 1e6);
                
                // 计算带宽
                long long bytes = 2 * total_elements * sizeof(float); // 输入和输出
                double bandwidth_gb_s = (bytes / (time_ms * 1e-3)) / 1e9;
                
                std::string shape_str = std::to_string(shape[0]);
                for (int i = 1; i < shape.size(); ++i) {
                    shape_str += "x" + std::to_string(shape[i]);
                }
                
                std::cout << std::setw(15) << shape_str 
                         << std::setw(12) << std::fixed << std::setprecision(2) << time_ms << " ms"
                         << std::setw(12) << std::fixed << std::setprecision(2) << gflops << " GFLOPS"
                         << std::setw(12) << std::fixed << std::setprecision(2) << bandwidth_gb_s << " GB/s"
                         << std::endl;
                
                results_.push_back({"ReLU", shape_str, 
                                  time_ms, gflops, bandwidth_gb_s, true});
                
            } catch (const std::exception& e) {
                std::cout << "ReLU " << " 失败: " << e.what() << std::endl;
                results_.push_back({"ReLU", "unknown", 0, 0, 0, false});
            }
        }
    }
    
    void RunSoftmaxBenchmark() {
        std::cout << "\n=== Softmax 性能基准测试 ===" << std::endl;
        
        std::vector<std::pair<int, int>> shapes = {
            {32, 1000}, {64, 1000}, {128, 1000}, {256, 1000}
        };
        
        for (auto [batch_size, num_classes] : shapes) {
            try {
                // 创建张量
                Tensor<float> input({static_cast<size_t>(batch_size), static_cast<size_t>(num_classes)});
                Tensor<float> output({static_cast<size_t>(batch_size), static_cast<size_t>(num_classes)});
                
                // 初始化数据
                int total_elements = batch_size * num_classes;
                std::vector<float> h_input(total_elements);
                for (int i = 0; i < total_elements; ++i) {
                    h_input[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 10.0f;
                }
                
                cudaMemcpy(input.data(), h_input.data(), input.bytes(), cudaMemcpyHostToDevice);
                
                // 创建Softmax算子
                Softmax<float> softmax;
                
                // 预热
                for (int i = 0; i < 3; ++i) {
                    softmax.Forward(input, output, 1);
                }
                cudaDeviceSynchronize();
                
                // 性能测试
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < 100; ++i) {
                    softmax.Forward(input, output, 1);
                }
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                double time_ms = duration.count() / 1000.0 / 100.0; // 平均每次执行时间
                
                // 计算GFLOPS (Softmax涉及exp, sum, div操作)
                long long flops = total_elements * 5; // 近似计算
                double gflops = flops / (time_ms * 1e6);
                
                // 计算带宽
                long long bytes = 2 * total_elements * sizeof(float);
                double bandwidth_gb_s = (bytes / (time_ms * 1e-3)) / 1e9;
                
                std::string shape_str = std::to_string(batch_size) + "x" + std::to_string(num_classes);
                
                std::cout << std::setw(12) << shape_str 
                         << std::setw(12) << std::fixed << std::setprecision(2) << time_ms << " ms"
                         << std::setw(12) << std::fixed << std::setprecision(2) << gflops << " GFLOPS"
                         << std::setw(12) << std::fixed << std::setprecision(2) << bandwidth_gb_s << " GB/s"
                         << std::endl;
                
                results_.push_back({"Softmax", shape_str, 
                                  time_ms, gflops, bandwidth_gb_s, true});
                
            } catch (const std::exception& e) {
                std::cout << "Softmax " << batch_size << "x" << num_classes << " 失败: " << e.what() << std::endl;
                results_.push_back({"Softmax", std::to_string(batch_size) + "x" + std::to_string(num_classes), 
                                  0, 0, 0, false});
            }
        }
    }
    
    void PrintSummary() {
        std::cout << "\n=== 性能测试总结 ===" << std::endl;
        std::cout << std::setw(12) << "操作" << std::setw(15) << "尺寸" 
                 << std::setw(12) << "时间(ms)" << std::setw(12) << "GFLOPS" 
                 << std::setw(12) << "带宽(GB/s)" << std::setw(8) << "状态" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        for (const auto& result : results_) {
            std::cout << std::setw(12) << result.operation 
                     << std::setw(15) << result.size
                     << std::setw(12) << std::fixed << std::setprecision(2) << result.time_ms
                     << std::setw(12) << std::fixed << std::setprecision(2) << result.gflops
                     << std::setw(12) << std::fixed << std::setprecision(2) << result.bandwidth_gb_s
                     << std::setw(8) << (result.success ? "成功" : "失败") << std::endl;
        }
    }
    
    void RunAllBenchmarks() {
        std::cout << "=== cuOP 性能基准测试 ===" << std::endl;
        std::cout << "开始时间: " << std::chrono::system_clock::now().time_since_epoch().count() << std::endl;
        
        RunGemmBenchmark();
        RunGemvBenchmark();
        RunDotBenchmark();
        RunReluBenchmark();
        RunSoftmaxBenchmark();
        
        PrintSummary();
    }
};

int main() {
    // 初始化CUDA
    cudaSetDevice(0);
    cudaFree(0);
    
    PerformanceBenchmark benchmark;
    benchmark.RunAllBenchmarks();
    
    return 0;
}
