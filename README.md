# cuOP

![C++17](https://img.shields.io/badge/C++-17-blue)
![CUDA](https://img.shields.io/badge/CUDA-11.5%2B-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![CMake](https://img.shields.io/badge/CMake-3.20.x-red)
![JIT](https://img.shields.io/badge/JIT-Enabled-orange)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![pybind11](https://img.shields.io/badge/pybind11-2.10%2B-green)
![Repo Size](https://img.shields.io/github/repo-size/Cyxuan0311/cuOP)

高性能 CUDA 算子与内存管理库，支持高效的张量运算、内存池优化、JIT实时编译优化和 HPC/深度学习常用算子。提供完整的 C++ 和 Python API 接口。

---

## 🚀 主要特性

- **🚀 高性能CUDA算子**: 支持GEMM、GEMV、ReLU、Softmax、MatMul等常用算子
- **⚡ JIT实时编译**: 智能包装器模式，零侵入性的运行时内核优化，支持自动调优和硬件特性利用
- **💾 持久化缓存**: 编译结果持久化，显著提升重复使用性能（25x-67x加速）
- **🧠 智能内存管理**: 多级缓存、内存碎片整理、智能预分配，减少频繁 cudaMalloc/cudaFree 带来的性能损耗
- **🔧 灵活配置**: 可自定义JIT配置和优化参数
- **📊 性能分析**: 内置性能分析和基准测试工具
- **🐍 Python原生**: 完全Python化的API设计，易于使用
- **📈 性能基准**: 提供标准化的算子性能测试和JIT优化效果对比

## 📁 目录结构

```
cuOP/
├── include/                # 头文件目录
│   ├── base/               # 基础设施（如内存池）
│   ├── cuda_op/            # CUDA 算子接口与实现
│   │   ├── abstract/       # 算子抽象接口
│   │   └── detail/         # 具体算子实现（cuBlas/cuDNN）
│   │       ├── cuBlas/     # GEMM/GEMV 等 BLAS 算子
│   │       └── cuDNN/      # ReLU 等深度学习算子
│   ├── data/               # 张量（Tensor）等数据结构
│   ├── util/               # 工具类（状态码等）
│   └── jit/                # JIT实时编译系统
│       ├── jit_config.hpp      # JIT配置系统
│       ├── ijit_plugin.hpp     # JIT插件接口
│       ├── jit_compiler.hpp    # JIT编译器
│       ├── jit_wrapper.hpp     # 智能包装器
│       └── jit_persistent_cache.hpp # 持久化缓存系统
├── src/                    # 源码实现
│   ├── base/               # 内存池等实现
│   ├── cuda_op/            # CUDA 算子实现
│   │   └── detail/         # 具体算子实现
│   ├── util/               # 工具类实现
│   └── jit/                # JIT系统实现
│       ├── jit_compiler.cu         # JIT编译器实现
│       ├── global_jit_manager.cu   # 全局JIT管理器
│       ├── jit_persistent_cache.cu # 持久化缓存实现
│       └── jit_docs.md             # JIT系统文档
├── python/                 # Python API接口
│   ├── setup.py            # Python包构建配置
│   ├── requirements.txt    # Python依赖包列表
│   ├── README.md           # Python API使用文档
│   ├── cuop/               # Python包源码
│   │   ├── __init__.py     # 主包初始化文件
│   │   └── core.cpp        # 核心Python绑定
│   └── examples/           # Python使用示例
│       ├── basic_usage.py      # 基本使用示例
│       └── advanced_usage.py   # 高级功能示例
├── bench/                  # 性能基准测试
│   ├── cuBlas/             # BLAS 算子基准
│   │   ├── gemm/           # GEMM 性能测试
│   │   └── gemv/           # GEMV 性能测试
│   └── cuDNN/              # cuDNN 算子基准
├── test/                   # 单元测试
│   ├── cuBlas/             # BLAS 算子测试
│   ├── cuDNN/              # cuDNN 算子测试
│   ├── JIT_test/           # JIT系统测试
│   │   ├── test_jit_system.cpp     # JIT系统测试程序
│   │   ├── test_persistent_cache.cpp # 持久化缓存测试
│   │   └── CMakeLists.txt          # JIT测试构建配置
│   └── util/               # 工具类测试
│       ├── test_status_code.cpp    # 错误码系统测试
│       └── CMakeLists.txt          # 工具测试构建配置
├── docs/                   # 项目文档
│   └── jit_persistent_cache_guide.md # JIT持久化缓存使用指南
├── third_party/            # 第三方依赖
├── CMakeLists.txt          # 顶层 CMake 构建脚本
├── .clang-format           # 代码风格配置
├── License                 # 许可证
└── README.md               # 项目说明
```

## 🔧 编译与运行

### 系统要求
- **CUDA**: 11.5 及以上（推荐 CUDA 12.x）
- **cuBLAS**: 如需 BLAS 算子
- **cuDNN**: 如需深度学习算子
- **NVRTC**: JIT实时编译
- **glog**: Google log库
- **GTest**: 单元测试
- **CMake**: 3.14 及以上 
- **GCC**: 7.0 及以上 
- **google benchmark**: 可选

### C++ 构建

```bash
# 克隆仓库
git clone https://github.com/Cyxuan0311/cuOP.git
cd cuOP

# 创建构建目录
mkdir build && cd build

# 配置和构建
cmake ..
make -j$(nproc)

# 安装（可选）
sudo make install
```

编译完成后会在项目根目录的 **lib** 目录下生成对应的动态库文件：
- `libcuop_cublas.so` - cuBLAS算子库
- `libcuop_cudnn.so` - cuDNN算子库  
- `libcuop_jit.so` - JIT实时编译库

### Python 构建

```bash
# 进入Python目录
cd python

# 安装依赖
pip install -r requirements.txt

# 构建Python包
pip install -e .

# 验证安装
python -c "import cuop; cuop.print_system_info()"
```

## 🧪 测试与基准

### 运行基准测试

```bash
# cuBLAS算子基准测试
./bench/cuBlas/gemm/bench_gemm
./bench/cuBlas/gemv/bench_gemv

# cuDNN算子基准测试
./bench/cuDNN/Relu/bench_relu
```

### 运行单元测试

```bash
# cuBLAS算子测试
./test/cuBlas/test_gemm/test_gemm
./test/cuBlas/test_gemv/test_gemv

# cuDNN算子测试
./test/cuDNN/test_relu/test_relu

# JIT系统测试
./test/JIT_test/test_jit_system
./test/JIT_test/test_persistent_cache

# 工具类测试
./test/util/test_status_code
```

## 💻 使用示例

### C++ 使用示例

#### JIT系统使用

```cpp
#include "jit/jit_wrapper.hpp"
#include "cuda_op/detail/cuBlas/gemm.hpp"

// 创建原始算子
Gemm<float> gemm;
gemm.SetWeight(weight);

// 创建JIT包装器（一行代码启用JIT优化）
JITWrapper<Gemm<float>> jit_gemm(gemm);
jit_gemm.EnableJIT(true);

// 启用持久化缓存
jit_gemm.EnablePersistentCache(true);
jit_gemm.SetPersistentCacheDirectory("./jit_cache");

// 使用方式与原始算子完全相同
jit_gemm.Forward(input, output);

// 获取性能信息
auto profile = jit_gemm.GetPerformanceProfile();
std::cout << "执行时间: " << profile.execution_time << "ms" << std::endl;
std::cout << "GFLOPS: " << profile.gflops << std::endl;
```

#### 基础算子使用

```cpp
#include "cuda_op/detail/cuBlas/gemm.hpp"
#include "data/tensor.hpp"

// 创建张量
Tensor<float> input({1000, 1000});
Tensor<float> weight({1000, 1000});
Tensor<float> output({1000, 1000});

// 使用GEMM算子
Gemm<float> gemm;
gemm.SetWeight(weight);
gemm.Forward(input, output);
```

### Python 使用示例

#### 基本使用

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

#### JIT优化使用

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

#### 批量处理

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

## 📊 性能特性

### 🚀 **内存池优化**
- **多级缓存**: 线程本地、全局、预分配缓存
- **智能管理**: 自动内存碎片整理和过期缓存清理
- **性能提升**: 显著减少内存分配开销

### 🔧 **JIT实时编译**
- **零侵入性**: 包装器模式，现有代码完全不变
- **智能优化**: 多级内核架构，自动选择最优配置
- **硬件感知**: 充分利用Tensor Core、TMA等最新特性
- **持久化缓存**: 编译结果持久化，重复使用性能提升25x-67x

### 📊 **算子支持**
- **cuBLAS**: GEMM、GEMV等高性能BLAS算子
- **cuDNN**: Conv、Pool、ReLU等深度学习算子
- **扩展性**: 插件化架构，易于添加新算子

### 🐍 **Python API**
- **原生性能**: 直接调用C++核心，无Python开销
- **易用性**: 完全Python化的API设计
- **功能完整**: 支持cuOP的所有核心功能
- **生产就绪**: 完善的错误处理和文档

## 🚨 常见问题

### C++ 相关问题

- **no CUDA-capable device is detected**
  - 请确保已正确安装 CUDA 驱动，并且 `ldd ./bench_gemm | grep cuda` 能看到 libcudart.so 路径。

- **链接错误**
  - 请检查 CMakeLists.txt 是否正确链接了 cudart、cublas、cudnn、nvrtc、glog 等依赖。

- **JIT编译失败**
  - 检查NVRTC是否正确安装，确保CUDA版本支持JIT编译。
  - 查看日志输出，JIT系统会自动回退到原始算子执行。

### Python 相关问题

- **ImportError: cuOP core module failed to import**
  - 确保CUDA环境正确配置
  - 检查pybind11是否正确安装
  - 验证包是否正确构建

- **CUDA内存不足**
  - 使用 `cuop.empty_cache()` 清理缓存
  - 检查GPU内存使用情况
  - 考虑使用较小的张量或批量处理

## 📚 文档

- **JIT系统**: 详细文档请参考 `src/jit/jit_docs.md`
- **持久化缓存**: 使用指南请参考 `docs/jit_persistent_cache_guide.md`
- **Python API**: 完整文档请参考 `python/README.md`
- **错误码系统**: 参考 `include/util/status_code.hpp`

## 🤝 贡献

欢迎提交 issue 和 PR！建议先阅读代码结构和注释。

### 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/your-username/cuOP.git
cd cuOP

# 安装开发依赖
pip install -r python/requirements.txt

# 构建项目
mkdir build && cd build
cmake ..
make -j$(nproc)

# 运行测试
make test
```

## 📄 License

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## 📞 支持

如果您遇到问题或有建议，请：

1. 查看 [文档](https://cuop.readthedocs.io/)
2. 搜索 [Issues](https://github.com/Cyxuan0311/cuOP/issues)
3. 创建新的 [Issue](https://github.com/Cyxuan0311/cuOP/issues/new)
4. 加入 [Discussions](https://github.com/Cyxuan0311/cuOP/discussions)

## 🎯 路线图

- [x] 基础CUDA算子支持
- [x] JIT实时编译系统
- [x] 持久化缓存优化
- [x] Python API接口
- [x] 错误码系统优化
- [ ] 更多算子支持
- [ ] 分布式计算支持
- [ ] 更多硬件架构支持
- [ ] 性能监控和调优工具

---

**cuOP** - 让CUDA计算更简单、更高效！ 🚀
