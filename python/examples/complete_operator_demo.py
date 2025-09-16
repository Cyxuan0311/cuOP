#!/usr/bin/env python3
"""
cuOP 完整算子演示

本示例展示了cuOP库中所有已实现算子的使用方法，包括cuBlas和cuDNN算子。
"""

import numpy as np
import cuop

def demo_cublas_operators():
    """演示cuBlas算子"""
    print("=== cuBlas算子演示 ===")
    
    # 创建测试数据
    n = 1000
    x = cuop.tensor(np.random.randn(n).astype(np.float32))
    y = cuop.tensor(np.random.randn(n).astype(np.float32))
    z = cuop.zeros([n])
    
    print(f"向量大小: {n}")
    
    # SCAL: 标量向量乘法
    print("\n1. SCAL - 标量向量乘法")
    scal = cuop.ScalFloat(2.5)
    scal.forward(x)
    print(f"SCAL完成: x = 2.5 * x")
    
    # AXPY: 向量加法
    print("\n2. AXPY - 向量加法")
    axpy = cuop.AxpyFloat(1.5)
    axpy.forward(x, y)
    print(f"AXPY完成: y = 1.5 * x + y")
    
    # COPY: 向量复制
    print("\n3. COPY - 向量复制")
    copy = cuop.CopyFloat()
    copy.forward(x, z)
    print(f"COPY完成: z = x")
    
    # DOT: 向量点积
    print("\n4. DOT - 向量点积")
    dot = cuop.DotFloat()
    result = dot.forward(x, y)
    print(f"DOT完成: result = x^T * y = {result}")
    
    # 矩阵运算
    m = 512
    A = cuop.tensor(np.random.randn(m, m).astype(np.float32))
    B = cuop.tensor(np.random.randn(m, m).astype(np.float32))
    C = cuop.zeros([m, m])
    
    print(f"\n矩阵大小: {m}x{m}")
    
    # GEMM: 通用矩阵乘法
    print("\n5. GEMM - 通用矩阵乘法")
    gemm = cuop.GemmFloat()
    gemm.set_weight(B)
    gemm.forward(A, C)
    print(f"GEMM完成: C = A * B")
    
    # GEMV: 矩阵向量乘法
    print("\n6. GEMV - 矩阵向量乘法")
    v = cuop.tensor(np.random.randn(m).astype(np.float32))
    w = cuop.zeros([m])
    gemv = cuop.GemvFloat()
    gemv.set_weight(A)
    gemv.forward(v, w)
    print(f"GEMV完成: w = A * v")
    
    # SYMM: 对称矩阵乘法
    print("\n7. SYMM - 对称矩阵乘法")
    # 创建对称矩阵
    S = cuop.tensor(np.random.randn(m, m).astype(np.float32))
    S_data = S.to_numpy()
    S_data = (S_data + S_data.T) / 2  # 使矩阵对称
    S = cuop.tensor(S_data)
    
    symm = cuop.SymmFloat(0, 1, 1.0, 0.0)  # left, upper, alpha=1.0, beta=0.0
    symm.set_weight(S)
    symm.forward(B, C)
    print(f"SYMM完成: C = S * B (S为对称矩阵)")
    
    # TRSM: 三角矩阵求解
    print("\n8. TRSM - 三角矩阵求解")
    # 创建下三角矩阵
    L = cuop.tensor(np.random.randn(m, m).astype(np.float32))
    L_data = L.to_numpy()
    L_data = np.tril(L_data)  # 下三角
    np.fill_diagonal(L_data, np.abs(np.diag(L_data)) + 1)  # 确保对角元素不为0
    L = cuop.tensor(L_data)
    
    trsm = cuop.TrsmFloat(0, 1, 0, 0, 1.0)  # left, lower, non-trans, non-unit, alpha=1.0
    trsm.set_matrix_a(L)
    trsm.forward(B, B)
    print(f"TRSM完成: 求解 L * X = B，结果存储在B中")

def demo_cudnn_operators():
    """演示cuDNN算子"""
    print("\n=== cuDNN算子演示 ===")
    
    # 创建测试数据
    batch_size, channels, height, width = 32, 64, 224, 224
    input_4d = cuop.tensor(np.random.randn(batch_size, channels, height, width).astype(np.float32))
    output_4d = cuop.zeros([batch_size, channels, height, width])
    
    print(f"4D张量形状: {batch_size}x{channels}x{height}x{width}")
    
    # ReLU激活
    print("\n1. ReLU - 激活函数")
    relu = cuop.ReluFloat()
    relu.forward(input_4d, output_4d)
    print(f"ReLU完成: output = max(0, input)")
    
    # Softmax归一化
    print("\n2. Softmax - 归一化")
    # 创建2D输入用于softmax
    input_2d = cuop.tensor(np.random.randn(1000, 10).astype(np.float32))
    output_2d = cuop.zeros([1000, 10])
    softmax = cuop.SoftmaxFloat()
    softmax.forward(input_2d, output_2d, axis=-1)
    print(f"Softmax完成: 在最后一个维度上计算softmax")
    
    # BatchNorm批归一化
    print("\n3. BatchNorm - 批归一化")
    batchnorm = cuop.BatchNormFloat()
    gamma = cuop.ones([channels])
    beta = cuop.zeros([channels])
    running_mean = cuop.zeros([channels])
    running_var = cuop.ones([channels])
    batchnorm.set_gamma(gamma)
    batchnorm.set_beta(beta)
    batchnorm.set_running_mean(running_mean)
    batchnorm.set_running_var(running_var)
    batchnorm.forward(input_4d, output_4d)
    print(f"BatchNorm完成: 批归一化处理")
    
    # LayerNorm层归一化
    print("\n4. LayerNorm - 层归一化")
    layernorm = cuop.LayerNormFloat()
    layernorm.set_gamma(gamma)
    layernorm.set_beta(beta)
    layernorm.forward(input_4d, output_4d)
    print(f"LayerNorm完成: 层归一化处理")
    
    # Convolution2D卷积
    print("\n5. Convolution2D - 2D卷积")
    in_channels, out_channels = 64, 128
    kernel_h, kernel_w = 3, 3
    stride_h, stride_w = 1, 1
    conv = cuop.Convolution2DFloat(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w)
    weight = cuop.tensor(np.random.randn(out_channels, in_channels, kernel_h, kernel_w).astype(np.float32))
    bias = cuop.tensor(np.random.randn(out_channels).astype(np.float32))
    conv.set_weight(weight)
    conv.set_bias(bias)
    conv_output = cuop.zeros([batch_size, out_channels, height, width])
    conv.forward(input_4d, conv_output, pad_h=1, pad_w=1)
    print(f"Convolution2D完成: {in_channels} -> {out_channels} 通道卷积")
    
    # MatMul矩阵乘法
    print("\n6. MatMul - 矩阵乘法")
    A = cuop.tensor(np.random.randn(1000, 1000).astype(np.float32))
    B = cuop.tensor(np.random.randn(1000, 1000).astype(np.float32))
    C = cuop.zeros([1000, 1000])
    matmul = cuop.MatMulFloat()
    matmul.forward(A, B, C, axis=-1)
    print(f"MatMul完成: C = A * B")
    
    # BatchMatMul批量矩阵乘法
    print("\n7. BatchMatMul - 批量矩阵乘法")
    batch_size_mat = 32
    A_batch = cuop.tensor(np.random.randn(batch_size_mat, 256, 256).astype(np.float32))
    B_batch = cuop.tensor(np.random.randn(batch_size_mat, 256, 256).astype(np.float32))
    C_batch = cuop.zeros([batch_size_mat, 256, 256])
    batch_matmul = cuop.BatchMatMulFloat()
    batch_matmul.forward(A_batch, B_batch, C_batch)
    print(f"BatchMatMul完成: 批量矩阵乘法 {batch_size_mat} 个 256x256 矩阵")
    
    # 池化层
    print("\n8. 池化层操作")
    pool_input = cuop.tensor(np.random.randn(32, 64, 56, 56).astype(np.float32))
    pool_output = cuop.zeros([32, 64, 28, 28])
    
    # MaxPool2D
    maxpool = cuop.MaxPool2DFloat()
    maxpool.forward(pool_input, pool_output, kernel_h=2, kernel_w=2)
    print(f"MaxPool2D完成: 2x2最大池化")
    
    # AveragePool2D
    avgpool = cuop.AveragePool2DFloat()
    avgpool.forward(pool_input, pool_output, kernel_h=2, kernel_w=2)
    print(f"AveragePool2D完成: 2x2平均池化")
    
    # GlobalMaxPool2D
    global_maxpool = cuop.GlobalMaxPool2DFloat()
    global_pool_output = cuop.zeros([32, 64, 1, 1])
    global_maxpool.forward(pool_input, global_pool_output, kernel_h=56, kernel_w=56)
    print(f"GlobalMaxPool2D完成: 全局最大池化")
    
    # GlobalAveragePool2D
    global_avgpool = cuop.GlobalAveragePool2DFloat()
    global_avgpool.forward(pool_input, global_pool_output, kernel_h=56, kernel_w=56)
    print(f"GlobalAveragePool2D完成: 全局平均池化")
    
    # 张量操作
    print("\n9. 张量操作")
    
    # Flatten
    flatten = cuop.FlattenFloat()
    flatten_output = cuop.zeros([32, 64*56*56])
    flatten.forward(pool_input, flatten_output, start_dim=1)
    print(f"Flatten完成: 张量展平")
    
    # View
    view = cuop.ViewFloat()
    view_output = cuop.zeros([32, 64, 56, 56])
    view.set_offset([0, 0, 0, 0])
    view.set_shape([32, 64, 56, 56])
    view.forward(pool_input, view_output, [0, 0, 0, 0], [32, 64, 56, 56])
    print(f"View完成: 张量视图操作")

def demo_jit_optimization():
    """演示JIT优化"""
    print("\n=== JIT优化演示 ===")
    
    # 创建测试数据
    A = cuop.tensor(np.random.randn(1024, 1024).astype(np.float32))
    B = cuop.tensor(np.random.randn(1024, 1024).astype(np.float32))
    C = cuop.zeros([1024, 1024])
    
    print(f"矩阵大小: 1024x1024")
    
    # 创建基础算子
    gemm = cuop.GemmFloat()
    gemm.set_weight(B)
    
    # 创建JIT包装器
    jit_gemm = cuop.JITGemmFloat(gemm)
    
    # 配置JIT
    jit_gemm.enable_jit(True)
    jit_gemm.enable_persistent_cache(True)
    jit_gemm.set_persistent_cache_directory("./jit_cache")
    
    # 设置JIT配置
    config = cuop.get_default_jit_config()
    config.kernel_type = "tiled"
    config.tile_size = 32
    config.block_size = 256
    config.enable_tensor_core = True
    jit_gemm.set_jit_config(config)
    
    print("\nJIT配置:")
    print(f"  - 内核类型: {config.kernel_type}")
    print(f"  - 瓦片大小: {config.tile_size}")
    print(f"  - 块大小: {config.block_size}")
    print(f"  - Tensor Core: {config.enable_tensor_core}")
    
    # 执行计算
    print("\n执行JIT优化的GEMM...")
    jit_gemm.forward(A, C)
    
    # 获取性能信息
    profile = jit_gemm.get_performance_profile()
    print(f"\n性能信息:")
    print(f"  - 执行时间: {profile.execution_time:.3f} ms")
    print(f"  - GFLOPS: {profile.gflops:.2f}")
    print(f"  - 吞吐量: {profile.throughput:.2f} GB/s")
    print(f"  - 内核类型: {profile.kernel_type}")
    
    # 检查JIT编译状态
    is_compiled = jit_gemm.is_jit_compiled()
    print(f"  - JIT已编译: {is_compiled}")

def demo_performance_benchmark():
    """演示性能基准测试"""
    print("\n=== 性能基准测试 ===")
    
    # 测试不同大小的矩阵
    sizes = [256, 512, 1024, 2048]
    
    for size in sizes:
        print(f"\n测试矩阵大小: {size}x{size}")
        
        # 创建测试数据
        A = cuop.tensor(np.random.randn(size, size).astype(np.float32))
        B = cuop.tensor(np.random.randn(size, size).astype(np.float32))
        C = cuop.zeros([size, size])
        
        # 创建算子
        gemm = cuop.GemmFloat()
        gemm.set_weight(B)
        
        # 运行基准测试
        stats = cuop.benchmark(gemm, A, C, num_runs=10, warmup_runs=3)
        
        print(f"  平均时间: {stats['mean_time_ms']:.3f} ms")
        print(f"  标准差: {stats['std_time_ms']:.3f} ms")
        print(f"  最小时间: {stats['min_time_ms']:.3f} ms")
        print(f"  最大时间: {stats['max_time_ms']:.3f} ms")

def main():
    """主函数"""
    print("cuOP 完整算子演示")
    print("=" * 50)
    
    # 检查CUDA可用性
    if not cuop.is_cuda_available():
        print("错误: CUDA不可用，请检查CUDA环境")
        return
    
    # 打印系统信息
    cuop.print_system_info()
    
    try:
        # 演示cuBlas算子
        demo_cublas_operators()
        
        # 演示cuDNN算子
        demo_cudnn_operators()
        
        # 演示JIT优化
        demo_jit_optimization()
        
        # 演示性能基准测试
        demo_performance_benchmark()
        
        print("\n" + "=" * 50)
        print("所有演示完成！")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
