#!/usr/bin/env python3
"""
cuOP Python API 基本使用示例

这个示例展示了如何使用cuOP Python API进行基本的矩阵运算和JIT优化。
"""

import numpy as np
import time
import cuop

def main():
    print("=== cuOP Python API 基本使用示例 ===\n")
    
    # 1. 检查系统信息
    print("1. 系统信息检查:")
    cuop.print_system_info()
    print()
    
    # 2. 创建张量
    print("2. 创建张量:")
    shape = [1000, 1000]
    
    # 从numpy数组创建
    a = cuop.tensor(np.random.randn(*shape).astype(np.float32))
    b = cuop.tensor(np.random.randn(*shape).astype(np.float32))
    c = cuop.zeros(shape)
    
    print(f"   - 输入张量 A: {a.shape}, dtype: {a.dtype}")
    print(f"   - 权重张量 B: {b.shape}, dtype: {b.dtype}")
    print(f"   - 输出张量 C: {c.shape}, dtype: {c.dtype}")
    print()
    
    # 3. 基础算子使用
    print("3. 基础算子使用:")
    
    # GEMM算子
    gemm = cuop.GemmFloat()
    gemm.set_weight(b)
    
    start_time = time.perf_counter()
    gemm.forward(a, c)
    cuop.synchronize()
    end_time = time.perf_counter()
    
    print(f"   - GEMM执行时间: {(end_time - start_time) * 1000:.3f} ms")
    print()
    
    # 4. JIT优化算子使用
    print("4. JIT优化算子使用:")
    
    # 创建JIT优化的GEMM算子
    jit_gemm = cuop.JITGemmFloat(gemm)
    
    # 启用JIT优化
    jit_gemm.enable_jit(True)
    
    # 启用持久化缓存
    jit_gemm.enable_persistent_cache(True)
    jit_gemm.set_persistent_cache_directory("./jit_cache")
    
    # 设置JIT配置
    config = cuop.get_default_jit_config()
    config.kernel_type = "tiled"
    config.tile_size = 32
    config.block_size = 256
    config.optimization_level = "O2"
    config.enable_tensor_core = True
    
    jit_gemm.set_jit_config(config)
    
    print(f"   - JIT配置: {config}")
    print()
    
    # 5. 性能对比测试
    print("5. 性能对比测试:")
    
    # 预热运行
    for _ in range(5):
        jit_gemm.forward(a, c)
    cuop.synchronize()
    
    # 基础算子测试
    times_basic = []
    for _ in range(10):
        start_time = time.perf_counter()
        gemm.forward(a, c)
        cuop.synchronize()
        end_time = time.perf_counter()
        times_basic.append((end_time - start_time) * 1000)
    
    # JIT算子测试
    times_jit = []
    for _ in range(10):
        start_time = time.perf_counter()
        jit_gemm.forward(a, c)
        cuop.synchronize()
        end_time = time.perf_counter()
        times_jit.append((end_time - start_time) * 1000)
    
    # 计算统计信息
    basic_mean = np.mean(times_basic)
    jit_mean = np.mean(times_jit)
    speedup = basic_mean / jit_mean if jit_mean > 0 else float('inf')
    
    print(f"   - 基础算子平均时间: {basic_mean:.3f} ms")
    print(f"   - JIT算子平均时间: {jit_mean:.3f} ms")
    print(f"   - 加速比: {speedup:.2f}x")
    print()
    
    # 6. 获取性能分析信息
    print("6. 性能分析信息:")
    profile = jit_gemm.get_performance_profile()
    print(f"   - 执行时间: {profile.execution_time:.3f} ms")
    print(f"   - 内核类型: {profile.kernel_type}")
    print(f"   - 矩阵大小: {profile.matrix_size}")
    print(f"   - 吞吐量: {profile.throughput:.2f} GB/s")
    print(f"   - GFLOPS: {profile.gflops:.2f}")
    print()
    
    # 7. 其他算子示例
    print("7. 其他算子示例:")
    
    # ReLU算子
    relu = cuop.ReluFloat()
    relu_output = cuop.zeros(shape)
    
    start_time = time.perf_counter()
    relu.forward(a, relu_output)
    cuop.synchronize()
    end_time = time.perf_counter()
    
    print(f"   - ReLU执行时间: {(end_time - start_time) * 1000:.3f} ms")
    
    # Softmax算子
    softmax = cuop.SoftmaxFloat()
    softmax_output = cuop.zeros(shape)
    
    start_time = time.perf_counter()
    softmax.forward(a, softmax_output)
    cuop.synchronize()
    end_time = time.perf_counter()
    
    print(f"   - Softmax执行时间: {(end_time - start_time) * 1000:.3f} ms")
    print()
    
    # 8. 便捷函数使用
    print("8. 便捷函数使用:")
    
    # 创建随机张量
    random_tensor = cuop.random([500, 500], dtype='float32', mean=0.0, std=1.0)
    print(f"   - 随机张量: {random_tensor.shape}")
    
    # 创建单位矩阵
    identity = cuop.eye(100, dtype='float32')
    print(f"   - 单位矩阵: {identity.shape}")
    
    # 创建线性空间
    linspace_tensor = cuop.linspace(0, 1, 100, dtype='float32')
    print(f"   - 线性空间: {linspace_tensor.shape}")
    print()
    
    # 9. 内存管理
    print("9. 内存管理:")
    
    # 获取内存信息
    free_mem, total_mem = cuop.get_memory_info()
    print(f"   - 总内存: {total_mem / (1024**3):.2f} GB")
    print(f"   - 可用内存: {free_mem / (1024**3):.2f} GB")
    print(f"   - 内存使用率: {(1 - free_mem / total_mem) * 100:.1f}%")
    print()
    
    # 10. 结果验证
    print("10. 结果验证:")
    
    # 将结果转换为numpy数组
    result_numpy = c.to_numpy()
    
    # 计算numpy版本的参考结果
    a_numpy = a.to_numpy()
    b_numpy = b.to_numpy()
    reference = np.dot(a_numpy, b_numpy)
    
    # 计算误差
    error = np.abs(result_numpy - reference).max()
    print(f"   - 最大误差: {error:.2e}")
    
    if error < 1e-5:
        print("   - ✅ 结果正确")
    else:
        print("   - ❌ 结果有误")
    
    print("\n=== 示例完成 ===")

def benchmark_example():
    """性能基准测试示例"""
    print("\n=== 性能基准测试示例 ===\n")
    
    # 创建不同大小的测试数据
    sizes = [512, 1024, 2048, 4096]
    
    for size in sizes:
        print(f"矩阵大小: {size}x{size}")
        
        # 创建测试数据
        shape = [size, size]
        a = cuop.tensor(np.random.randn(*shape).astype(np.float32))
        b = cuop.tensor(np.random.randn(*shape).astype(np.float32))
        c = cuop.zeros(shape)
        
        # 创建算子
        gemm = cuop.GemmFloat()
        gemm.set_weight(b)
        
        jit_gemm = cuop.JITGemmFloat(gemm)
        jit_gemm.enable_jit(True)
        jit_gemm.enable_persistent_cache(True)
        
        # 运行基准测试
        basic_stats = cuop.benchmark(gemm, a, c, num_runs=20, warmup_runs=5)
        jit_stats = cuop.benchmark(jit_gemm, a, c, num_runs=20, warmup_runs=5)
        
        # 计算加速比
        speedup = basic_stats["mean_time_ms"] / jit_stats["mean_time_ms"]
        
        print(f"  基础算子: {basic_stats['mean_time_ms']:.3f} ± {basic_stats['std_time_ms']:.3f} ms")
        print(f"  JIT算子:  {jit_stats['mean_time_ms']:.3f} ± {jit_stats['std_time_ms']:.3f} ms")
        print(f"  加速比:   {speedup:.2f}x")
        print()

def jit_config_example():
    """JIT配置示例"""
    print("\n=== JIT配置示例 ===\n")
    
    # 获取默认配置
    default_config = cuop.get_default_jit_config()
    print("默认JIT配置:")
    print(f"  - 启用JIT: {default_config.enable_jit}")
    print(f"  - 内核类型: {default_config.kernel_type}")
    print(f"  - 瓦片大小: {default_config.tile_size}")
    print(f"  - 块大小: {default_config.block_size}")
    print(f"  - 优化级别: {default_config.optimization_level}")
    print(f"  - 启用Tensor Core: {default_config.enable_tensor_core}")
    print(f"  - 启用TMA: {default_config.enable_tma}")
    print()
    
    # 创建自定义配置
    custom_config = cuop.JITConfig()
    custom_config.enable_jit = True
    custom_config.kernel_type = "warp_optimized"
    custom_config.tile_size = 64
    custom_config.block_size = 512
    custom_config.optimization_level = "O3"
    custom_config.enable_tensor_core = True
    custom_config.enable_tma = False
    custom_config.max_registers = 128
    custom_config.enable_shared_memory_opt = True
    custom_config.enable_loop_unroll = True
    custom_config.enable_memory_coalescing = True
    
    print("自定义JIT配置:")
    print(f"  - 启用JIT: {custom_config.enable_jit}")
    print(f"  - 内核类型: {custom_config.kernel_type}")
    print(f"  - 瓦片大小: {custom_config.tile_size}")
    print(f"  - 块大小: {custom_config.block_size}")
    print(f"  - 优化级别: {custom_config.optimization_level}")
    print(f"  - 启用Tensor Core: {custom_config.enable_tensor_core}")
    print(f"  - 启用TMA: {custom_config.enable_tma}")
    print(f"  - 最大寄存器数: {custom_config.max_registers}")
    print(f"  - 启用共享内存优化: {custom_config.enable_shared_memory_opt}")
    print(f"  - 启用循环展开: {custom_config.enable_loop_unroll}")
    print(f"  - 启用内存合并: {custom_config.enable_memory_coalescing}")
    print()

if __name__ == "__main__":
    try:
        # 运行基本示例
        main()
        
        # 运行基准测试示例
        benchmark_example()
        
        # 运行JIT配置示例
        jit_config_example()
        
    except Exception as e:
        print(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc() 