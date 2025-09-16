"""
cuOP - High-performance CUDA operator and memory management library

A Python package providing high-performance CUDA operators with JIT optimization,
persistent caching, and efficient memory management.

Example:
    >>> import cuop
    >>> import numpy as np
    >>> 
    >>> # Create tensors
    >>> a = cuop.tensor(np.random.randn(1000, 1000).astype(np.float32))
    >>> b = cuop.tensor(np.random.randn(1000, 1000).astype(np.float32))
    >>> c = cuop.zeros([1000, 1000])
    >>> 
    >>> # Use GEMM operator with JIT optimization
    >>> gemm = cuop.JITGemmFloat()
    >>> gemm.enable_jit(True)
    >>> gemm.enable_persistent_cache(True)
    >>> gemm.set_persistent_cache_directory("./jit_cache")
    >>> 
    >>> gemm.set_weight(b)
    >>> gemm.forward(a, c)
    >>> 
    >>> # Get performance information
    >>> profile = gemm.get_performance_profile()
    >>> print(f"Execution time: {profile.execution_time}ms")
    >>> print(f"GFLOPS: {profile.gflops}")
    >>> 
    >>> # Convert back to numpy
    >>> result = c.to_numpy()
"""

# 版本信息
__version__ = "0.1.0"
__author__ = "cuOP Team"
__email__ = "contact@cuop.dev"

# 导入核心功能
try:
    from .core import (
        # Tensor类
        TensorFloat, TensorDouble, TensorInt,
        
        # cuBlas算子
        ScalFloat, ScalDouble,
        AxpyFloat, AxpyDouble,
        CopyFloat, CopyDouble,
        DotFloat, DotDouble,
        GemmFloat, GemmDouble,
        GemvFloat, GemvDouble,
        SymmFloat, SymmDouble,
        TrsmFloat, TrsmDouble,
        
        # cuDNN算子
        ReluFloat, ReluDouble,
        SoftmaxFloat, SoftmaxDouble,
        BatchNormFloat, BatchNormDouble,
        LayerNormFloat, LayerNormDouble,
        Convolution2DFloat, Convolution2DDouble,
        MatMulFloat, MatMulDouble,
        BatchMatMulFloat, BatchMatMulDouble,
        FlattenFloat, FlattenDouble,
        ViewFloat, ViewDouble,
        MaxPool2DFloat, MaxPool2DDouble,
        AveragePool2DFloat, AveragePool2DDouble,
        GlobalMaxPool2DFloat, GlobalMaxPool2DDouble,
        GlobalAveragePool2DFloat, GlobalAveragePool2DDouble,
        
        # JIT包装器
        JITGemmFloat, JITGemmDouble,
        JITGemvFloat, JITGemvDouble,
        JITReluFloat, JITReluDouble,
        JITSoftmaxFloat, JITSoftmaxDouble,
        JITBatchNormFloat, JITBatchNormDouble,
        JITLayerNormFloat, JITLayerNormDouble,
        JITConvolution2DFloat, JITConvolution2DDouble,
        JITMatMulFloat, JITMatMulDouble,
        
        # JIT配置
        JITConfig, GlobalJITConfig,
        
        # 性能分析
        PerformanceProfile,
        
        # 工具函数
        tensor, randn, ones, zeros,
        get_device_count, set_device, get_device,
        synchronize, get_memory_info, empty_cache,
        
        # 异常类
        CUDAError, MemoryError, CompilationError, ExecutionError,
    )
    
    # 便捷别名
    Tensor = TensorFloat  # 默认使用float类型
    
    # cuBlas算子别名
    Scal = ScalFloat
    Axpy = AxpyFloat
    Copy = CopyFloat
    Dot = DotFloat
    Gemm = GemmFloat
    Gemv = GemvFloat
    Symm = SymmFloat
    Trsm = TrsmFloat
    
    # cuDNN算子别名
    Relu = ReluFloat
    Softmax = SoftmaxFloat
    BatchNorm = BatchNormFloat
    LayerNorm = LayerNormFloat
    Convolution2D = Convolution2DFloat
    MatMul = MatMulFloat
    BatchMatMul = BatchMatMulFloat
    Flatten = FlattenFloat
    View = ViewFloat
    MaxPool2D = MaxPool2DFloat
    AveragePool2D = AveragePool2DFloat
    GlobalMaxPool2D = GlobalMaxPool2DFloat
    GlobalAveragePool2D = GlobalAveragePool2DFloat
    
    # JIT包装器别名
    JITGemm = JITGemmFloat
    JITGemv = JITGemvFloat
    JITRelu = JITReluFloat
    JITSoftmax = JITSoftmaxFloat
    JITBatchNorm = JITBatchNormFloat
    JITLayerNorm = JITLayerNormFloat
    JITConvolution2D = JITConvolution2DFloat
    JITMatMul = JITMatMulFloat
    
    # 标记导入成功
    _import_success = True
    
except ImportError as e:
    _import_success = False
    _import_error = str(e)
    
    # 提供友好的错误信息
    def _import_error_wrapper(*args, **kwargs):
        raise ImportError(
            f"cuOP core module failed to import: {_import_error}\n"
            "Please ensure that:\n"
            "1. CUDA toolkit is properly installed\n"
            "2. pybind11 is available\n"
            "3. The package is built correctly"
        ) from e
    
    # 创建占位符类
    class _Placeholder:
        def __init__(self, *args, **kwargs):
            _import_error_wrapper()
        
        def __getattr__(self, name):
            _import_error_wrapper()
    
    # 设置占位符
    Tensor = _Placeholder
    
    # cuBlas算子占位符
    Scal = _Placeholder
    Axpy = _Placeholder
    Copy = _Placeholder
    Dot = _Placeholder
    Gemm = _Placeholder
    Gemv = _Placeholder
    Symm = _Placeholder
    Trsm = _Placeholder
    
    # cuDNN算子占位符
    Relu = _Placeholder
    Softmax = _Placeholder
    BatchNorm = _Placeholder
    LayerNorm = _Placeholder
    Convolution2D = _Placeholder
    MatMul = _Placeholder
    BatchMatMul = _Placeholder
    Flatten = _Placeholder
    View = _Placeholder
    MaxPool2D = _Placeholder
    AveragePool2D = _Placeholder
    GlobalMaxPool2D = _Placeholder
    GlobalAveragePool2D = _Placeholder
    
    # JIT包装器占位符
    JITGemm = _Placeholder
    JITGemv = _Placeholder
    JITRelu = _Placeholder
    JITSoftmax = _Placeholder
    JITBatchNorm = _Placeholder
    JITLayerNorm = _Placeholder
    JITConvolution2D = _Placeholder
    JITMatMul = _Placeholder

# 检查CUDA可用性
def is_cuda_available():
    """检查CUDA是否可用"""
    if not _import_success:
        return False
    
    try:
        count = get_device_count()
        return count > 0
    except:
        return False

# 获取CUDA信息
def get_cuda_info():
    """获取CUDA环境信息"""
    if not _import_success:
        return {
            "available": False,
            "error": _import_error
        }
    
    try:
        device_count = get_device_count()
        current_device = get_device()
        
        # 获取内存信息
        free_mem, total_mem = get_memory_info()
        
        return {
            "available": True,
            "device_count": device_count,
            "current_device": current_device,
            "free_memory": free_mem,
            "total_memory": total_mem,
            "free_memory_gb": free_mem / (1024**3),
            "total_memory_gb": total_mem / (1024**3),
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

# 便捷函数：创建随机张量
def random(shape, dtype='float32', mean=0.0, std=1.0):
    """创建随机张量
    
    Args:
        shape: 张量形状
        dtype: 数据类型 ('float32', 'float64', 'int32')
        mean: 均值
        std: 标准差
    
    Returns:
        Tensor: 随机张量
    """
    if dtype == 'float32':
        return randn(shape, mean, std)
    elif dtype == 'float64':
        return randn(shape, float(mean), float(std))
    elif dtype == 'int32':
        # 对于整数类型，使用均匀分布
        import numpy as np
        arr = np.random.randint(int(mean - std), int(mean + std), shape, dtype=np.int32)
        return tensor(arr)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

# 便捷函数：创建单位矩阵
def eye(n, dtype='float32'):
    """创建单位矩阵
    
    Args:
        n: 矩阵大小
        dtype: 数据类型
    
    Returns:
        Tensor: 单位矩阵
    """
    import numpy as np
    arr = np.eye(n, dtype=dtype)
    return tensor(arr)

# 便捷函数：创建线性空间
def linspace(start, stop, num, dtype='float32'):
    """创建线性空间
    
    Args:
        start: 起始值
        stop: 结束值
        num: 元素数量
        dtype: 数据类型
    
    Returns:
        Tensor: 线性空间张量
    """
    import numpy as np
    arr = np.linspace(start, stop, num, dtype=dtype)
    return tensor(arr)

# 便捷函数：创建对数空间
def logspace(start, stop, num, base=10.0, dtype='float32'):
    """创建对数空间
    
    Args:
        start: 起始指数
        stop: 结束指数
        num: 元素数量
        base: 底数
        dtype: 数据类型
    
    Returns:
        Tensor: 对数空间张量
    """
    import numpy as np
    arr = np.logspace(start, stop, num, base=base, dtype=dtype)
    return tensor(arr)

# 便捷函数：创建网格
def meshgrid(*xi, indexing='xy'):
    """创建网格坐标
    
    Args:
        *xi: 坐标向量
        indexing: 索引顺序 ('xy' 或 'ij')
    
    Returns:
        list: 网格坐标张量列表
    """
    import numpy as np
    grids = np.meshgrid(*xi, indexing=indexing)
    return [tensor(grid) for grid in grids]

# 便捷函数：创建随机种子
def seed(seed_value):
    """设置随机种子
    
    Args:
        seed_value: 种子值
    """
    import numpy as np
    np.random.seed(seed_value)

# 便捷函数：获取默认JIT配置
def get_default_jit_config():
    """获取默认JIT配置
    
    Returns:
        JITConfig: 默认配置
    """
    if not _import_success:
        _import_error_wrapper()
    
    config = JITConfig()
    config.enable_jit = True
    config.kernel_type = "tiled"
    config.tile_size = 32
    config.block_size = 256
    config.optimization_level = "O2"
    config.enable_tensor_core = True
    config.enable_tma = True
    config.max_registers = 255
    config.enable_shared_memory_opt = True
    config.enable_loop_unroll = True
    config.enable_memory_coalescing = True
    
    return config

# 便捷函数：获取默认全局JIT配置
def get_default_global_jit_config():
    """获取默认全局JIT配置
    
    Returns:
        GlobalJITConfig: 默认全局配置
    """
    if not _import_success:
        _import_error_wrapper()
    
    config = GlobalJITConfig()
    config.enable_jit = True
    config.enable_auto_tuning = True
    config.enable_caching = True
    config.cache_dir = "./jit_cache"
    config.max_cache_size = 10 * 1024 * 1024 * 1024  # 10GB
    config.compilation_timeout = 30
    config.enable_tensor_core = True
    config.enable_tma = True
    config.max_compilation_threads = 4
    config.enable_debug = False
    
    return config

# 便捷函数：创建JIT优化的算子
def create_jit_operator(operator_type, dtype='float32', **kwargs):
    """创建JIT优化的算子
    
    Args:
        operator_type: 算子类型 ('gemm', 'gemv', 'relu', 'softmax', 'matmul', 'batchnorm', 'layernorm', 'convolution2d')
        dtype: 数据类型
        **kwargs: JIT配置参数
    
    Returns:
        JIT包装的算子
    """
    if not _import_success:
        _import_error_wrapper()
    
    # 创建基础算子
    if operator_type.lower() == 'gemm':
        if dtype == 'float32':
            base_op = GemmFloat()
            jit_op = JITGemmFloat(base_op)
        elif dtype == 'float64':
            base_op = GemmDouble()
            jit_op = JITGemmDouble(base_op)
        else:
            raise ValueError(f"Unsupported dtype for GEMM: {dtype}")
    
    elif operator_type.lower() == 'gemv':
        if dtype == 'float32':
            base_op = GemvFloat()
            jit_op = JITGemvFloat(base_op)
        elif dtype == 'float64':
            base_op = GemvDouble()
            jit_op = JITGemvDouble(base_op)
        else:
            raise ValueError(f"Unsupported dtype for GEMV: {dtype}")
    
    elif operator_type.lower() == 'relu':
        if dtype == 'float32':
            base_op = ReluFloat()
            jit_op = JITReluFloat(base_op)
        elif dtype == 'float64':
            base_op = ReluDouble()
            jit_op = JITReluDouble(base_op)
        else:
            raise ValueError(f"Unsupported dtype for ReLU: {dtype}")
    
    elif operator_type.lower() == 'softmax':
        if dtype == 'float32':
            base_op = SoftmaxFloat()
            jit_op = JITSoftmaxFloat(base_op)
        elif dtype == 'float64':
            base_op = SoftmaxDouble()
            jit_op = JITSoftmaxDouble(base_op)
        else:
            raise ValueError(f"Unsupported dtype for Softmax: {dtype}")
    
    elif operator_type.lower() == 'matmul':
        if dtype == 'float32':
            base_op = MatMulFloat()
            jit_op = JITMatMulFloat(base_op)
        elif dtype == 'float64':
            base_op = MatMulDouble()
            jit_op = JITMatMulDouble(base_op)
        else:
            raise ValueError(f"Unsupported dtype for MatMul: {dtype}")
    
    elif operator_type.lower() == 'batchnorm':
        if dtype == 'float32':
            base_op = BatchNormFloat()
            jit_op = JITBatchNormFloat(base_op)
        elif dtype == 'float64':
            base_op = BatchNormDouble()
            jit_op = JITBatchNormDouble(base_op)
        else:
            raise ValueError(f"Unsupported dtype for BatchNorm: {dtype}")
    
    elif operator_type.lower() == 'layernorm':
        if dtype == 'float32':
            base_op = LayerNormFloat()
            jit_op = JITLayerNormFloat(base_op)
        elif dtype == 'float64':
            base_op = LayerNormDouble()
            jit_op = JITLayerNormDouble(base_op)
        else:
            raise ValueError(f"Unsupported dtype for LayerNorm: {dtype}")
    
    elif operator_type.lower() == 'convolution2d':
        if dtype == 'float32':
            base_op = Convolution2DFloat()
            jit_op = JITConvolution2DFloat(base_op)
        elif dtype == 'float64':
            base_op = Convolution2DDouble()
            jit_op = JITConvolution2DDouble(base_op)
        else:
            raise ValueError(f"Unsupported dtype for Convolution2D: {dtype}")
    
    else:
        raise ValueError(f"Unsupported operator type: {operator_type}")
    
    # 应用JIT配置
    jit_op.enable_jit(True)
    jit_op.enable_persistent_cache(True)
    
    # 应用自定义配置
    for key, value in kwargs.items():
        if hasattr(jit_op, key):
            setattr(jit_op, key, value)
    
    return jit_op

# 便捷函数：性能基准测试
def benchmark(operator, input_tensor, output_tensor, num_runs=100, warmup_runs=10):
    """运行性能基准测试
    
    Args:
        operator: 要测试的算子
        input_tensor: 输入张量
        output_tensor: 输出张量
        num_runs: 测试运行次数
        warmup_runs: 预热运行次数
    
    Returns:
        dict: 性能统计信息
    """
    if not _import_success:
        _import_error_wrapper()
    
    import time
    
    # 预热运行
    for _ in range(warmup_runs):
        operator.forward(input_tensor, output_tensor)
    
    synchronize()
    
    # 实际测试
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        operator.forward(input_tensor, output_tensor)
        synchronize()
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    # 计算统计信息
    times = sorted(times)
    mean_time = sum(times) / len(times)
    median_time = times[len(times) // 2]
    min_time = times[0]
    max_time = times[-1]
    
    # 计算标准差
    variance = sum((t - mean_time) ** 2 for t in times) / len(times)
    std_time = variance ** 0.5
    
    return {
        "mean_time_ms": mean_time,
        "median_time_ms": median_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "std_time_ms": std_time,
        "num_runs": num_runs,
        "warmup_runs": warmup_runs,
    }

# 便捷函数：打印系统信息
def print_system_info():
    """打印系统信息"""
    print("=== cuOP System Information ===")
    print(f"Version: {__version__}")
    print(f"Import Success: {_import_success}")
    
    if _import_success:
        cuda_info = get_cuda_info()
        if cuda_info["available"]:
            print(f"CUDA Available: Yes")
            print(f"Device Count: {cuda_info['device_count']}")
            print(f"Current Device: {cuda_info['current_device']}")
            print(f"Total Memory: {cuda_info['total_memory_gb']:.2f} GB")
            print(f"Free Memory: {cuda_info['free_memory_gb']:.2f} GB")
        else:
            print(f"CUDA Available: No")
            if "error" in cuda_info:
                print(f"Error: {cuda_info['error']}")
    else:
        print(f"CUDA Available: No")
        print(f"Import Error: {_import_error}")
    
    print("================================")

# 便捷函数：检查环境
def check_environment():
    """检查运行环境
    
    Returns:
        dict: 环境检查结果
    """
    result = {
        "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
        "cuop_version": __version__,
        "import_success": _import_success,
        "cuda_available": False,
        "issues": []
    }
    
    if not _import_success:
        result["issues"].append(f"Core module import failed: {_import_error}")
        return result
    
    try:
        cuda_info = get_cuda_info()
        result["cuda_available"] = cuda_info["available"]
        
        if cuda_info["available"]:
            result["device_count"] = cuda_info["device_count"]
            result["current_device"] = cuda_info["current_device"]
            result["total_memory_gb"] = cuda_info["total_memory_gb"]
            result["free_memory_gb"] = cuda_info["free_memory_gb"]
        else:
            result["issues"].append(f"CUDA not available: {cuda_info.get('error', 'Unknown error')}")
    
    except Exception as e:
        result["issues"].append(f"CUDA check failed: {str(e)}")
    
    return result

# 导出主要接口
__all__ = [
    # 版本信息
    "__version__", "__author__", "__email__",
    
    # 核心类
    "Tensor", "TensorFloat", "TensorDouble", "TensorInt",
    
    # cuBlas算子
    "Scal", "ScalFloat", "ScalDouble",
    "Axpy", "AxpyFloat", "AxpyDouble",
    "Copy", "CopyFloat", "CopyDouble",
    "Dot", "DotFloat", "DotDouble",
    "Gemm", "GemmFloat", "GemmDouble",
    "Gemv", "GemvFloat", "GemvDouble",
    "Symm", "SymmFloat", "SymmDouble",
    "Trsm", "TrsmFloat", "TrsmDouble",
    
    # cuDNN算子
    "Relu", "ReluFloat", "ReluDouble",
    "Softmax", "SoftmaxFloat", "SoftmaxDouble",
    "BatchNorm", "BatchNormFloat", "BatchNormDouble",
    "LayerNorm", "LayerNormFloat", "LayerNormDouble",
    "Convolution2D", "Convolution2DFloat", "Convolution2DDouble",
    "MatMul", "MatMulFloat", "MatMulDouble",
    "BatchMatMul", "BatchMatMulFloat", "BatchMatMulDouble",
    "Flatten", "FlattenFloat", "FlattenDouble",
    "View", "ViewFloat", "ViewDouble",
    "MaxPool2D", "MaxPool2DFloat", "MaxPool2DDouble",
    "AveragePool2D", "AveragePool2DFloat", "AveragePool2DDouble",
    "GlobalMaxPool2D", "GlobalMaxPool2DFloat", "GlobalMaxPool2DDouble",
    "GlobalAveragePool2D", "GlobalAveragePool2DFloat", "GlobalAveragePool2DDouble",
    
    # JIT包装器
    "JITGemm", "JITGemmFloat", "JITGemmDouble",
    "JITGemv", "JITGemvFloat", "JITGemvDouble",
    "JITRelu", "JITReluFloat", "JITReluDouble",
    "JITSoftmax", "JITSoftmaxFloat", "JITSoftmaxDouble",
    "JITBatchNorm", "JITBatchNormFloat", "JITBatchNormDouble",
    "JITLayerNorm", "JITLayerNormFloat", "JITLayerNormDouble",
    "JITConvolution2D", "JITConvolution2DFloat", "JITConvolution2DDouble",
    "JITMatMul", "JITMatMulFloat", "JITMatMulDouble",
    
    # 配置类
    "JITConfig", "GlobalJITConfig",
    
    # 性能分析
    "PerformanceProfile",
    
    # 工具函数
    "tensor", "randn", "ones", "zeros", "random",
    "eye", "linspace", "logspace", "meshgrid",
    "seed", "get_default_jit_config", "get_default_global_jit_config",
    "create_jit_operator", "benchmark",
    
    # 系统函数
    "is_cuda_available", "get_cuda_info", "print_system_info", "check_environment",
    "get_device_count", "set_device", "get_device",
    "synchronize", "get_memory_info", "empty_cache",
    
    # 异常类
    "CUDAError", "MemoryError", "CompilationError", "ExecutionError",
]

# 自动打印系统信息（可选）
if __name__ == "__main__":
    print_system_info() 