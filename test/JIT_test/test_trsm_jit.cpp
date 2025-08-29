#include "jit/jit_wrapper.hpp"
#include "cuda_op/detail/cuBlas/trsm.hpp"
#include "data/tensor.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace cu_op_mem;

// 创建下三角矩阵A
Tensor<float> CreateLowerTriangularMatrix(int size) {
    Tensor<float> A({size, size});
    A.Zero();
    
    // 填充下三角部分，确保对角元素非零
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j <= i; ++j) {
            if (i == j) {
                A[i * size + j] = 1.0f + (i + 1) * 0.1f;  // 对角元素
            } else {
                A[i * size + j] = (i + j + 1) * 0.1f;     // 下三角元素
            }
        }
    }
    
    return A;
}

// 验证TRSM结果
bool VerifyTrsmResult(const Tensor<float>& A, const Tensor<float>& B, 
                     const Tensor<float>& X, float alpha) {
    int m = A.shape()[0];
    int n = B.shape()[1];
    
    // 计算 A * X
    Tensor<float> AX({m, n});
    AX.Zero();
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < m; ++k) {
                sum += A[i * m + k] * X[k * n + j];
            }
            AX[i * n + j] = sum;
        }
    }
    
    // 计算 alpha * B
    Tensor<float> alphaB({m, n});
    for (int i = 0; i < m * n; ++i) {
        alphaB[i] = alpha * B[i];
    }
    
    // 比较 A * X 和 alpha * B
    float max_error = 0.0f;
    for (int i = 0; i < m * n; ++i) {
        float error = std::abs(AX[i] - alphaB[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "    最大误差: " << max_error << std::endl;
    return max_error < 1e-4f;
}

void TestTrsmJIT(int m, int n, const std::string& kernel_type) {
    std::cout << "\n=== TRSM JIT测试 (m=" << m << ", n=" << n << ", kernel=" << kernel_type << ") ===" << std::endl;
    
    // 创建测试数据
    Tensor<float> A = CreateLowerTriangularMatrix(m);
    Tensor<float> B({m, n});
    Tensor<float> X({m, n});
    
    // 初始化B矩阵
    B.Fill(1.0f);
    X.Zero();
    
    float alpha = 2.0f;
    
    std::cout << "1. 创建原始TRSM算子..." << std::endl;
    
    // 创建原始TRSM算子
    Trsm<float> original_trsm(0, 1, 0, 0, alpha);  // left, lower, no-trans, non-unit
    
    std::cout << "2. 创建JIT包装器..." << std::endl;
    
    // 创建JIT包装器
    JITWrapper<Trsm<float>> jit_trsm(original_trsm);
    jit_trsm.EnableJIT(true);
    
    // 配置JIT参数
    JITConfig config;
    config.kernel_type = kernel_type;
    config.tile_size = 16;
    config.block_size = 256;
    config.optimization_level = "O2";
    config.enable_tensor_core = false;
    config.enable_tma = false;
    
    jit_trsm.SetJITConfig(config);
    
    std::cout << "3. 编译JIT内核..." << std::endl;
    
    // 编译JIT内核
    auto compile_status = jit_trsm.CompileJIT();
    if (compile_status != StatusCode::SUCCESS) {
        std::cout << "   JIT编译失败，回退到原始算子" << std::endl;
        jit_trsm.EnableJIT(false);
    } else {
        std::cout << "   JIT编译成功" << std::endl;
    }
    
    std::cout << "4. 执行TRSM计算..." << std::endl;
    
    // 执行计算
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto status = jit_trsm.Forward(A, X);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double>(end_time - start_time).count();
    
    if (status == StatusCode::SUCCESS) {
        std::cout << "   执行成功，耗时: " << std::fixed << std::setprecision(3) 
                  << execution_time * 1000 << " ms" << std::endl;
        
        // 验证结果
        std::cout << "5. 验证结果..." << std::endl;
        if (VerifyTrsmResult(A, B, X, alpha)) {
            std::cout << "   结果验证通过 ✓" << std::endl;
        } else {
            std::cout << "   结果验证失败 ✗" << std::endl;
        }
        
        // 获取性能统计
        if (jit_trsm.IsJITEnabled()) {
            auto profile = jit_trsm.GetPerformanceProfile();
            std::cout << "   内核类型: " << profile.kernel_type << std::endl;
            std::cout << "   吞吐量: " << std::fixed << std::setprecision(2) 
                      << profile.throughput << " GFLOPS" << std::endl;
        }
        
    } else {
        std::cout << "   执行失败，错误码: " << static_cast<int>(status) << std::endl;
    }
}

int main() {
    // 初始化日志
    google::InitGoogleLogging("test_trsm_jit");
    google::SetStderrLogging(google::INFO);
    
    std::cout << "=== cuOP TRSM JIT System Test ===" << std::endl;
    
    try {
        // 初始化全局JIT管理器
        auto& global_manager = GlobalJITManager::Instance();
        auto status = global_manager.Initialize();
        if (status != StatusCode::SUCCESS) {
            std::cerr << "Failed to initialize Global JIT Manager" << std::endl;
            return -1;
        }
        
        // 设置全局配置
        GlobalJITConfig global_config;
        global_config.enable_jit = true;
        global_config.enable_auto_tuning = true;
        global_config.enable_caching = true;
        global_config.cache_dir = "./jit_cache";
        global_config.enable_debug = true;
        global_manager.SetGlobalConfig(global_config);
        
        // 注册BLAS JIT插件
        RegisterBlasJITPlugins();
        
        // 测试不同大小的矩阵
        std::vector<std::pair<int, int>> test_cases = {
            {64, 64},    // 小矩阵
            {256, 256},  // 中等矩阵
            {512, 512},  // 大矩阵
            {1024, 1024} // 超大矩阵
        };
        
        std::vector<std::string> kernel_types = {
            "basic",
            "tiled", 
            "warp_optimized",
            "blocked"
        };
        
        for (const auto& test_case : test_cases) {
            int m = test_case.first;
            int n = test_case.second;
            
            for (const auto& kernel_type : kernel_types) {
                TestTrsmJIT(m, n, kernel_type);
            }
        }
        
        std::cout << "\n=== 测试完成 ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "测试异常: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
} 