# cuOP Python Package

cuOP Python包提供了高性能CUDA算子的Python接口，支持JIT优化、持久化缓存和高效的内存管理。

## 快速安装

```bash
# 进入Python目录
cd python

# 安装依赖
pip install -r requirements.txt

# 构建并安装Python包
pip install -e .

# 验证安装
python -c "import cuop; cuop.print_system_info()"
```

## 快速开始

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

## 完整文档

详细的API文档和使用示例请参考：
- [Python API完整文档](../docs/PYTHON_API.md)

## 示例代码

- `examples/quick_start.py`: 快速开始示例
- `examples/complete_operator_demo.py`: 完整算子演示

## 系统要求

- Python 3.7+
- CUDA 11.0+
- NVIDIA GPU (支持CUDA)
- Linux/macOS/Windows

## 支持的算子

### cuBlas算子
- SCAL, AXPY, COPY, DOT, GEMM, GEMV, SYMM, TRSM

### cuDNN算子  
- ReLU, Softmax, BatchNorm, LayerNorm, Convolution2D, MatMul, BatchMatMul, Flatten, View, MaxPool2D, AveragePool2D, GlobalMaxPool2D, GlobalAveragePool2D

### JIT优化
- 所有主要算子都支持JIT优化
- 持久化缓存支持
- 自动性能调优

## 故障排除

如果遇到导入错误，请检查：
1. CUDA环境是否正确配置
2. pybind11是否正确安装
3. 包是否正确构建

更多问题请参考[完整文档](../docs/PYTHON_API.md)中的故障排除部分。