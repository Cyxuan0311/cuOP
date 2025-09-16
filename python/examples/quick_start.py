#!/usr/bin/env python3
"""
cuOP 快速开始示例

本示例展示了cuOP库的基本使用方法，包括最常用的算子。
"""

import numpy as np
import cuop

def main():
    """快速开始示例"""
    print("cuOP 快速开始示例")
    print("=" * 30)
    
    # 检查CUDA可用性
    if not cuop.is_cuda_available():
        print("错误: CUDA不可用，请检查CUDA环境")
        return
    
    # 打印系统信息
    cuop.print_system_info()
    
    # 1. 创建张量
    print("\n1. 创建张量")
    a = cuop.tensor(np.random.randn(1000, 1000).astype(np.float32))
    b = cuop.tensor(np.random.randn(1000, 1000).astype(np.float32))
    c = cuop.zeros([1000, 1000])
    print(f"创建了三个 1000x1000 的张量")
    
    # 2. 矩阵乘法 (GEMM)
    print("\n2. 矩阵乘法 (GEMM)")
    gemm = cuop.GemmFloat()
    gemm.set_weight(b)
    gemm.forward(a, c)
    print(f"GEMM完成: C = A * B")
    
    # 3. ReLU激活
    print("\n3. ReLU激活")
    relu = cuop.ReluFloat()
    relu.forward(a, a)  # 就地操作
    print(f"ReLU完成: A = max(0, A)")
    
    # 4. 向量运算
    print("\n4. 向量运算")
    x = cuop.tensor(np.random.randn(1000).astype(np.float32))
    y = cuop.tensor(np.random.randn(1000).astype(np.float32))
    
    # 标量乘法
    scal = cuop.ScalFloat(2.0)
    scal.forward(x)
    print(f"SCAL完成: x = 2.0 * x")
    
    # 向量加法
    axpy = cuop.AxpyFloat(1.5)
    axpy.forward(x, y)
    print(f"AXPY完成: y = 1.5 * x + y")
    
    # 向量点积
    dot = cuop.DotFloat()
    result = dot.forward(x, y)
    print(f"DOT完成: result = x^T * y = {result:.2f}")
    
    # 5. JIT优化
    print("\n5. JIT优化")
    jit_gemm = cuop.JITGemmFloat(gemm)
    jit_gemm.enable_jit(True)
    jit_gemm.enable_persistent_cache(True)
    
    # 设置JIT配置
    config = cuop.get_default_jit_config()
    config.kernel_type = "tiled"
    config.tile_size = 32
    jit_gemm.set_jit_config(config)
    
    # 执行JIT优化的计算
    jit_gemm.forward(a, c)
    profile = jit_gemm.get_performance_profile()
    print(f"JIT GEMM完成: {profile.execution_time:.3f} ms, {profile.gflops:.2f} GFLOPS")
    
    # 6. 转换为numpy
    print("\n6. 转换为numpy")
    result_numpy = c.to_numpy()
    print(f"结果形状: {result_numpy.shape}")
    print(f"结果统计: 均值={result_numpy.mean():.3f}, 标准差={result_numpy.std():.3f}")
    
    print("\n" + "=" * 30)
    print("快速开始示例完成！")

if __name__ == "__main__":
    main()
