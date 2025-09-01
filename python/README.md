# cuOP Python API

cuOP Python API 是一个高性能的CUDA算子和内存管理库，提供JIT优化、持久化缓存和高效的内存管理功能。

## 特性

- 🚀 **高性能CUDA算子**: 支持GEMM、GEMV、ReLU、Softmax、MatMul等常用算子
- ⚡ **JIT优化**: 实时内核优化，自动选择最佳配置
- 💾 **持久化缓存**: 编译结果持久化，显著提升重复使用性能
- 🧠 **智能内存管理**: 高效的内存池和自动内存优化
- 🔧 **灵活配置**: 可自定义JIT配置和优化参数
- 📊 **性能分析**: 内置性能分析和基准测试工具
- 🐍 **Python原生**: 完全Python化的API设计，易于使用

## 安装

### 系统要求

- Python 3.7+
- CUDA 11.0+
- NVIDIA GPU (支持CUDA)
- Linux/macOS/Windows

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/cuop/cuop.git
cd cuop

# 安装依赖
pip install -r requirements.txt

# 构建Python包
cd python
pip install -e .
```

### 依赖安装

```bash
pip install numpy pybind11
```

## 快速开始

### 基本使用

```python
import cuop
import numpy as np

# 创建张量
a = cuop.tensor(np.random.randn(1000, 1000).astype(np.float32))
b = cuop.tensor(np.random.randn(1000, 1000).astype(np.float32))
c = cuop.zeros([1000, 1000])

# 使用GEMM算子
gemm = cuop.GemmFloat()
gemm.set_weight(b)
gemm.forward(a, c)

# 转换为numpy数组
result = c.to_numpy()
```

### JIT优化使用

```python
# 创建JIT优化的算子
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

# 执行计算
jit_gemm.forward(a, c)

# 获取性能信息
profile = jit_gemm.get_performance_profile()
print(f"执行时间: {profile.execution_time}ms")
print(f"GFLOPS: {profile.gflops}")
```

### 批量处理

```python
# 创建批量数据
batch_size = 100
matrix_size = 256
batch_inputs = []
batch_weights = []

for i in range(batch_size):
    input_data = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    weight_data = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    
    batch_inputs.append(cuop.tensor(input_data))
    batch_weights.append(cuop.tensor(weight_data))

# 批量处理
gemm = cuop.GemmFloat()
batch_outputs = [cuop.zeros([matrix_size, matrix_size]) for _ in range(batch_size)]

for i in range(batch_size):
    gemm.set_weight(batch_weights[i])
    gemm.forward(batch_inputs[i], batch_outputs[i])

cuop.synchronize()
```

## API参考

### 核心类

#### Tensor

张量类，支持多种数据类型和形状。

```python
# 创建张量
tensor = cuop.TensorFloat([1000, 1000])
tensor = cuop.tensor(np.random.randn(1000, 1000).astype(np.float32))

# 张量操作
tensor.fill(1.0)           # 填充值
tensor.zero()              # 清零
tensor.copy_from(arr)      # 从numpy数组复制
result = tensor.to_numpy() # 转换为numpy数组

# 属性
shape = tensor.shape       # 获取形状
size = tensor.size         # 获取元素数量
dtype = tensor.dtype       # 获取数据类型
```

#### 算子类

支持多种CUDA算子：

- `GemmFloat/GemmDouble`: 矩阵乘法
- `GemvFloat/GemvDouble`: 矩阵向量乘法
- `ReluFloat/ReluDouble`: ReLU激活函数
- `SoftmaxFloat/SoftmaxDouble`: Softmax函数
- `MatMulFloat/MatMulDouble`: 矩阵乘法

```python
# 创建算子
gemm = cuop.GemmFloat()
gemm.set_weight(weight_tensor)

# 执行计算
gemm.forward(input_tensor, output_tensor)
```

#### JIT包装器

JIT优化的算子包装器：

```python
# 创建JIT包装器
jit_gemm = cuop.JITGemmFloat(gemm)

# 配置JIT
jit_gemm.enable_jit(True)
jit_gemm.enable_persistent_cache(True)
jit_gemm.set_persistent_cache_directory("./jit_cache")

# 设置配置
config = cuop.JITConfig()
config.kernel_type = "tiled"
config.tile_size = 32
config.block_size = 256
jit_gemm.set_jit_config(config)

# 执行
jit_gemm.forward(input_tensor, output_tensor)

# 获取性能信息
profile = jit_gemm.get_performance_profile()
```

### 配置类

#### JITConfig

JIT编译配置：

```python
config = cuop.JITConfig()
config.enable_jit = True
config.kernel_type = "tiled"           # 内核类型
config.tile_size = 32                  # 瓦片大小
config.block_size = 256                # 块大小
config.optimization_level = "O2"       # 优化级别
config.enable_tensor_core = True       # 启用Tensor Core
config.enable_tma = True               # 启用TMA
config.max_registers = 255             # 最大寄存器数
config.enable_shared_memory_opt = True # 启用共享内存优化
config.enable_loop_unroll = True       # 启用循环展开
config.enable_memory_coalescing = True # 启用内存合并
```

#### GlobalJITConfig

全局JIT配置：

```python
global_config = cuop.GlobalJITConfig()
global_config.enable_jit = True
global_config.enable_auto_tuning = True
global_config.enable_caching = True
global_config.cache_dir = "./jit_cache"
global_config.max_cache_size = 10 * 1024 * 1024 * 1024  # 10GB
global_config.compilation_timeout = 30
global_config.max_compilation_threads = 4
```

### 工具函数

#### 张量创建

```python
# 创建张量
tensor = cuop.tensor(numpy_array)
zeros = cuop.zeros(shape)
ones = cuop.ones(shape)
random_tensor = cuop.random(shape, dtype='float32', mean=0.0, std=1.0)

# 特殊张量
identity = cuop.eye(n, dtype='float32')
linspace_tensor = cuop.linspace(start, stop, num, dtype='float32')
logspace_tensor = cuop.logspace(start, stop, num, base=10.0, dtype='float32')
grid_tensors = cuop.meshgrid(*xi, indexing='xy')
```

#### 设备管理

```python
# 设备信息
device_count = cuop.get_device_count()
current_device = cuop.get_device()

# 设备控制
cuop.set_device(device_id)
cuop.synchronize()

# 内存信息
free_mem, total_mem = cuop.get_memory_info()
cuop.empty_cache()
```

#### 性能分析

```python
# 基准测试
stats = cuop.benchmark(operator, input_tensor, output_tensor, 
                       num_runs=100, warmup_runs=10)

print(f"平均时间: {stats['mean_time_ms']:.3f} ms")
print(f"标准差: {stats['std_time_ms']:.3f} ms")
print(f"最小时间: {stats['min_time_ms']:.3f} ms")
print(f"最大时间: {stats['max_time_ms']:.3f} ms")
```

## 最佳实践

### 1. 内存管理

```python
# 及时释放不需要的张量
del large_tensor
cuop.empty_cache()

# 使用上下文管理器管理内存
with cuop.memory_context():
    # 在这个上下文中创建的张量会自动管理
    tensor = cuop.tensor(data)
    # 使用tensor...
# 离开上下文后自动清理
```

### 2. JIT配置优化

```python
# 根据数据大小选择配置
if matrix_size < 1024:
    config.kernel_type = "simple"
    config.tile_size = 16
elif matrix_size < 4096:
    config.kernel_type = "tiled"
    config.tile_size = 32
else:
    config.kernel_type = "tensor_core"
    config.tile_size = 64
```

### 3. 批量处理

```python
# 使用批量处理提高效率
batch_size = 100
for i in range(0, total_size, batch_size):
    batch = data[i:i+batch_size]
    # 处理批次...
    cuop.synchronize()  # 定期同步
```

### 4. 错误处理

```python
try:
    result = operator.forward(input, output)
except cuop.MemoryError as e:
    print(f"内存不足: {e}")
    cuop.empty_cache()
except cuop.ExecutionError as e:
    print(f"执行错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

## 性能优化

### 1. 选择合适的算子

- 小矩阵 (< 512x512): 使用基础算子
- 中等矩阵 (512x512 - 4096x4096): 使用JIT优化算子
- 大矩阵 (> 4096x4096): 使用Tensor Core优化

### 2. 内存访问优化

- 使用连续的内存布局
- 避免频繁的内存分配/释放
- 利用共享内存优化

### 3. 并行化策略

- 根据GPU架构调整块大小
- 使用适当的瓦片大小
- 启用循环展开和向量化

## 故障排除

### 常见问题

1. **CUDA错误**
   ```python
   # 检查CUDA环境
   cuop.print_system_info()
   
   # 检查设备状态
   device_count = cuop.get_device_count()
   current_device = cuop.get_device()
   ```

2. **内存不足**
   ```python
   # 清理缓存
   cuop.empty_cache()
   
   # 检查内存使用
   free_mem, total_mem = cuop.get_memory_info()
   ```

3. **编译失败**
   ```python
   # 检查JIT配置
   config = cuop.get_default_jit_config()
   
   # 尝试不同的优化级别
   config.optimization_level = "O1"  # 降低优化级别
   ```

### 调试模式

```python
# 启用调试模式
global_config = cuop.get_default_global_jit_config()
global_config.enable_debug = True

# 获取详细错误信息
error_msg = operator.get_last_error()
```

## 示例代码

完整的示例代码请参考：

- `examples/basic_usage.py`: 基本使用示例
- `examples/advanced_usage.py`: 高级功能示例

## 贡献

欢迎贡献代码！请参考 [CONTRIBUTING.md](../CONTRIBUTING.md) 了解贡献指南。

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](../LICENSE) 文件。

## 支持

如果您遇到问题或有建议，请：

1. 查看 [文档](https://cuop.readthedocs.io/)
2. 搜索 [Issues](https://github.com/cuop/cuop/issues)
3. 创建新的 [Issue](https://github.com/cuop/cuop/issues/new)
4. 加入 [Discussions](https://github.com/cuop/cuop/discussions)

## 更新日志

### v0.1.0
- 初始版本发布
- 支持基础CUDA算子
- JIT优化系统
- 持久化缓存
- Python API 