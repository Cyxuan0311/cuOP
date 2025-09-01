#!/usr/bin/env python3
"""
cuOP Python API 高级使用示例

这个示例展示了cuOP Python API的高级功能，包括：
- 自定义JIT配置优化
- 批量处理
- 内存管理优化
- 错误处理
- 性能分析
"""

import numpy as np
import time
import cuop
from contextlib import contextmanager

@contextmanager
def timer(name):
    """计时器上下文管理器"""
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    print(f"{name}: {(end_time - start_time) * 1000:.3f} ms")

def memory_optimization_example():
    """内存优化示例"""
    print("=== 内存优化示例 ===\n")
    
    # 获取初始内存状态
    free_mem, total_mem = cuop.get_memory_info()
    print(f"初始状态:")
    print(f"  - 总内存: {total_mem / (1024**3):.2f} GB")
    print(f"  - 可用内存: {free_mem / (1024**3):.2f} GB")
    print(f"  - 使用率: {(1 - free_mem / total_mem) * 100:.1f}%")
    print()
    
    # 创建大张量
    large_shape = [5000, 5000]
    print(f"创建大张量: {large_shape}")
    
    with timer("创建张量"):
        a = cuop.tensor(np.random.randn(*large_shape).astype(np.float32))
        b = cuop.tensor(np.random.randn(*large_shape).astype(np.float32))
        c = cuop.zeros(large_shape)
    
    # 检查内存使用
    free_mem_after, _ = cuop.get_memory_info()
    used_mem = free_mem - free_mem_after
    print(f"  - 内存使用: {used_mem / (1024**3):.2f} GB")
    print()
    
    # 执行计算
    gemm = cuop.GemmFloat()
    gemm.set_weight(b)
    
    with timer("GEMM计算"):
        gemm.forward(a, c)
        cuop.synchronize()
    
    # 清理内存
    print("清理内存...")
    del a, b, c
    cuop.empty_cache()
    
    # 检查清理后的状态
    free_mem_clean, _ = cuop.get_memory_info()
    print(f"  - 清理后可用内存: {free_mem_clean / (1024**3):.2f} GB")
    print(f"  - 内存恢复: {(free_mem_clean - free_mem_after) / (1024**3):.2f} GB")
    print()

def batch_processing_example():
    """批量处理示例"""
    print("=== 批量处理示例 ===\n")
    
    # 创建批量数据
    batch_size = 100
    matrix_size = 256
    batch_shape = [batch_size, matrix_size, matrix_size]
    
    print(f"批量处理: {batch_shape}")
    
    # 创建批量输入和权重
    with timer("创建批量数据"):
        batch_inputs = []
        batch_weights = []
        
        for i in range(batch_size):
            # 为每个批次创建不同的数据
            input_data = np.random.randn(matrix_size, matrix_size).astype(np.float32)
            weight_data = np.random.randn(matrix_size, matrix_size).astype(np.float32)
            
            batch_inputs.append(cuop.tensor(input_data))
            batch_weights.append(cuop.tensor(weight_data))
    
    # 创建输出张量
    batch_outputs = [cuop.zeros([matrix_size, matrix_size]) for _ in range(batch_size)]
    
    # 创建算子
    gemm = cuop.GemmFloat()
    
    # 批量处理
    print("执行批量处理...")
    with timer("批量GEMM"):
        for i in range(batch_size):
            gemm.set_weight(batch_weights[i])
            gemm.forward(batch_inputs[i], batch_outputs[i])
        
        cuop.synchronize()
    
    print(f"  - 处理了 {batch_size} 个 {matrix_size}x{matrix_size} 矩阵")
    print()
    
    # 清理批量数据
    del batch_inputs, batch_weights, batch_outputs
    cuop.empty_cache()

def custom_jit_optimization_example():
    """自定义JIT优化示例"""
    print("=== 自定义JIT优化示例 ===\n")
    
    # 创建测试数据
    shape = [2048, 2048]
    a = cuop.tensor(np.random.randn(*shape).astype(np.float32))
    b = cuop.tensor(np.random.randn(*shape).astype(np.float32))
    c = cuop.zeros(shape)
    
    # 创建基础算子
    base_gemm = cuop.GemmFloat()
    base_gemm.set_weight(b)
    
    # 测试不同JIT配置的性能
    configs = [
        {
            "name": "基础配置",
            "config": cuop.get_default_jit_config()
        },
        {
            "name": "高性能配置",
            "config": cuop.JITConfig()
        },
        {
            "name": "内存优化配置",
            "config": cuop.JITConfig()
        }
    ]
    
    # 设置高性能配置
    configs[1]["config"].enable_jit = True
    configs[1]["config"].kernel_type = "tensor_core"
    configs[1]["config"].tile_size = 64
    configs[1]["config"].block_size = 512
    configs[1]["config"].optimization_level = "O3"
    configs[1]["config"].enable_tensor_core = True
    configs[1]["config"].enable_tma = True
    configs[1]["config"].max_registers = 255
    
    # 设置内存优化配置
    configs[2]["config"].enable_jit = True
    configs[2]["config"].kernel_type = "tiled"
    configs[2]["config"].tile_size = 32
    configs[2]["config"].block_size = 256
    configs[2]["config"].optimization_level = "O2"
    configs[2]["config"].enable_tensor_core = False
    configs[2]["config"].enable_tma = False
    configs[2]["config"].max_registers = 64
    configs[2]["config"].enable_shared_memory_opt = True
    
    # 测试每种配置
    results = []
    
    for config_info in configs:
        name = config_info["name"]
        config = config_info["config"]
        
        print(f"测试配置: {name}")
        print(f"  - 内核类型: {config.kernel_type}")
        print(f"  - 瓦片大小: {config.tile_size}")
        print(f"  - 块大小: {config.block_size}")
        print(f"  - 优化级别: {config.optimization_level}")
        print(f"  - 启用Tensor Core: {config.enable_tensor_core}")
        print(f"  - 最大寄存器数: {config.max_registers}")
        
        # 创建JIT算子
        jit_gemm = cuop.JITGemmFloat(base_gemm)
        jit_gemm.enable_jit(True)
        jit_gemm.enable_persistent_cache(True)
        jit_gemm.set_jit_config(config)
        
        # 预热
        for _ in range(3):
            jit_gemm.forward(a, c)
        cuop.synchronize()
        
        # 性能测试
        times = []
        for _ in range(10):
            start_time = time.perf_counter()
            jit_gemm.forward(a, c)
            cuop.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"  - 平均时间: {mean_time:.3f} ± {std_time:.3f} ms")
        print()
        
        results.append({
            "name": name,
            "mean_time": mean_time,
            "std_time": std_time,
            "config": config
        })
    
    # 分析结果
    print("性能分析结果:")
    best_result = min(results, key=lambda x: x["mean_time"])
    print(f"  - 最佳配置: {best_result['name']}")
    print(f"  - 最佳时间: {best_result['mean_time']:.3f} ms")
    
    for result in results:
        if result != best_result:
            speedup = result["mean_time"] / best_result["mean_time"]
            print(f"  - {result['name']} vs 最佳: {speedup:.2f}x")
    
    print()
    
    # 清理
    del a, b, c

def error_handling_example():
    """错误处理示例"""
    print("=== 错误处理示例 ===\n")
    
    try:
        # 尝试创建过大的张量
        huge_shape = [100000, 100000]
        print(f"尝试创建超大张量: {huge_shape}")
        
        huge_tensor = cuop.zeros(huge_shape)
        print("✅ 成功创建超大张量")
        
    except cuop.MemoryError as e:
        print(f"❌ 内存错误: {e}")
    except Exception as e:
        print(f"❌ 其他错误: {e}")
    
    print()
    
    try:
        # 尝试使用未初始化的算子
        print("尝试使用未初始化的算子...")
        
        gemm = cuop.GemmFloat()
        # 不设置权重就调用forward
        input_tensor = cuop.tensor(np.random.randn(100, 100).astype(np.float32))
        output_tensor = cuop.zeros([100, 100])
        
        gemm.forward(input_tensor, output_tensor)
        print("✅ 算子执行成功")
        
    except cuop.ExecutionError as e:
        print(f"❌ 执行错误: {e}")
    except Exception as e:
        print(f"❌ 其他错误: {e}")
    
    print()
    
    try:
        # 尝试使用不兼容的数据类型
        print("尝试使用不兼容的数据类型...")
        
        float_tensor = cuop.tensor(np.random.randn(100, 100).astype(np.float32))
        int_tensor = cuop.tensor(np.random.randint(0, 10, (100, 100)).astype(np.int32))
        
        # 这应该会失败，因为GEMM不支持混合类型
        gemm = cuop.GemmFloat()
        gemm.set_weight(int_tensor)
        gemm.forward(float_tensor, int_tensor)
        print("✅ 混合类型操作成功")
        
    except cuop.ValidationError as e:
        print(f"❌ 验证错误: {e}")
    except Exception as e:
        print(f"❌ 其他错误: {e}")
    
    print()

def performance_profiling_example():
    """性能分析示例"""
    print("=== 性能分析示例 ===\n")
    
    # 创建不同大小的测试数据
    sizes = [512, 1024, 2048, 4096]
    
    print("性能分析结果:")
    print(f"{'大小':<8} {'基础(ms)':<12} {'JIT(ms)':<12} {'加速比':<10} {'GFLOPS':<12}")
    print("-" * 60)
    
    for size in sizes:
        shape = [size, size]
        
        # 创建测试数据
        a = cuop.tensor(np.random.randn(*shape).astype(np.float32))
        b = cuop.tensor(np.random.randn(*shape).astype(np.float32))
        c = cuop.zeros(shape)
        
        # 基础算子
        gemm = cuop.GemmFloat()
        gemm.set_weight(b)
        
        # 预热
        for _ in range(3):
            gemm.forward(a, c)
        cuop.synchronize()
        
        # 测试基础算子
        basic_times = []
        for _ in range(5):
            start_time = time.perf_counter()
            gemm.forward(a, c)
            cuop.synchronize()
            end_time = time.perf_counter()
            basic_times.append((end_time - start_time) * 1000)
        
        basic_mean = np.mean(basic_times)
        
        # JIT算子
        jit_gemm = cuop.JITGemmFloat(gemm)
        jit_gemm.enable_jit(True)
        jit_gemm.enable_persistent_cache(True)
        
        # 预热
        for _ in range(3):
            jit_gemm.forward(a, c)
        cuop.synchronize()
        
        # 测试JIT算子
        jit_times = []
        for _ in range(5):
            start_time = time.perf_counter()
            jit_gemm.forward(a, c)
            cuop.synchronize()
            end_time = time.perf_counter()
            jit_times.append((end_time - start_time) * 1000)
        
        jit_mean = np.mean(jit_times)
        
        # 计算加速比和GFLOPS
        speedup = basic_mean / jit_mean if jit_mean > 0 else float('inf')
        
        # 计算GFLOPS (2 * M * N * K / time_ms / 1e6)
        operations = 2 * size * size * size
        gflops = operations / jit_mean / 1e6 if jit_mean > 0 else 0
        
        print(f"{size:<8} {basic_mean:<12.3f} {jit_mean:<12.3f} {speedup:<10.2f} {gflops:<12.2f}")
        
        # 清理
        del a, b, c
    
    print("-" * 60)
    print()
    
    # 清理缓存
    cuop.empty_cache()

def memory_pool_example():
    """内存池示例"""
    print("=== 内存池示例 ===\n")
    
    # 模拟内存分配和释放
    print("模拟内存分配和释放...")
    
    # 获取初始状态
    free_mem, total_mem = cuop.get_memory_info()
    print(f"初始可用内存: {free_mem / (1024**3):.2f} GB")
    
    # 创建多个张量
    tensors = []
    shapes = [
        [1000, 1000],
        [2000, 2000],
        [3000, 3000],
        [4000, 4000],
        [5000, 5000]
    ]
    
    print("分配内存...")
    for i, shape in enumerate(shapes):
        tensor = cuop.tensor(np.random.randn(*shape).astype(np.float32))
        tensors.append(tensor)
        
        current_free, _ = cuop.get_memory_info()
        used = free_mem - current_free
        print(f"  张量 {i+1} ({shape[0]}x{shape[1]}): 使用 {used / (1024**3):.2f} GB")
    
    # 检查最终状态
    final_free, _ = cuop.get_memory_info()
    total_used = free_mem - final_free
    print(f"总内存使用: {total_used / (1024**3):.2f} GB")
    
    # 释放部分张量
    print("\n释放部分张量...")
    for i in [0, 2, 4]:  # 释放第1、3、5个张量
        del tensors[i]
        tensors[i] = None
    
    # 清理缓存
    cuop.empty_cache()
    
    # 检查释放后的状态
    after_free, _ = cuop.get_memory_info()
    recovered = after_free - final_free
    print(f"内存恢复: {recovered / (1024**3):.2f} GB")
    
    # 释放所有张量
    print("\n释放所有张量...")
    del tensors
    cuop.empty_cache()
    
    # 最终状态
    final_free, _ = cuop.get_memory_info()
    print(f"最终可用内存: {final_free / (1024**3):.2f} GB")
    print()

def main():
    """主函数"""
    print("=== cuOP Python API 高级使用示例 ===\n")
    
    try:
        # 检查环境
        env_info = cuop.check_environment()
        if env_info["issues"]:
            print("环境检查发现问题:")
            for issue in env_info["issues"]:
                print(f"  - {issue}")
            print()
        
        # 运行各种示例
        memory_optimization_example()
        batch_processing_example()
        custom_jit_optimization_example()
        error_handling_example()
        performance_profiling_example()
        memory_pool_example()
        
        print("=== 所有高级示例完成 ===")
        
    except Exception as e:
        print(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 