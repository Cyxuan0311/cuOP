# cuOP

![C++17](https://img.shields.io/badge/C++-17-blue)
![CUDA](https://img.shields.io/badge/CUDA-11.5%2B-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![CMake](https://img.shields.io/badge/CMake-3.20.x-red)
![JIT](https://img.shields.io/badge/JIT-Enabled-orange)
![Repo Size](https://img.shields.io/github/repo-size/Cyxuan0311/cuOP)

高性能 CUDA 算子与内存管理库，支持高效的张量运算、内存池优化、JIT实时编译优化和 HPC/深度学习常用算子。

---

## 目录结构

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
│       └── jit_wrapper.hpp     # 智能包装器
├── src/                    # 源码实现
│   ├── base/               # 内存池等实现
│   ├── cuda_op/            # CUDA 算子实现
│   │   └── detail/         # 具体算子实现
│   ├── util/               # 工具类实现
│   └── jit/                # JIT系统实现
│       ├── jit_compiler.cu         # JIT编译器实现
│       ├── global_jit_manager.cu   # 全局JIT管理器
│       └── jit_docs.md             # JIT系统文档
├── bench/                  # 性能基准测试
│   ├── cuBlas/             # BLAS 算子基准
│   │   ├── gemm/           # GEMM 性能测试
│   │   └── gemv/           # GEMV 性能测试
│   └── cuDNN/              # cuDNN 算子基准
├── test/                   # 单元测试
│   ├── cuBlas/             # BLAS 算子测试
│   ├── cuDNN/              # cuDNN 算子测试
│   └── JIT_test/           # JIT系统测试
│       ├── test_jit_system.cpp     # JIT系统测试程序
│       └── CMakeLists.txt          # JIT测试构建配置
├── third_party/            # 第三方依赖
├── CMakeLists.txt          # 顶层 CMake 构建脚本
├── .clang-format           # 代码风格配置
├── License                 # 许可证
└── README.md               # 项目说明
```

## 主要功能

- **Tensor 管理**：高效多维张量分配与释放，基于自定义 CUDA 内存池。
- **算子支持**：
  - cuBLAS：GEMM（矩阵乘法）、GEMV（矩阵-向量乘法）等 BLAS 算子
  - cuDNN：Conv、MaxPool、SoftMax 等深度学习常用算子
- **内存池优化**：多级缓存、内存碎片整理、智能预分配，减少频繁 cudaMalloc/cudaFree 带来的性能损耗。
- **JIT实时编译**：智能包装器模式，零侵入性的运行时内核优化，支持自动调优和硬件特性利用。
- **性能基准**：提供标准化的算子性能测试和JIT优化效果对比。

## 编译与运行

### 依赖
- CUDA 11.5 及以上（推荐 CUDA 12.x）
- cuBLAS（如需 BLAS 算子）
- cuDNN（如需深度学习算子）
- NVRTC（JIT实时编译）
- glog (google log库)
- GTest（单元测试）
- CMake 3.14 及以上 
- GCC 7.0 及以上 
- google benchmark（可选）

### 构建示例
```bash
git clone https://github.com/Cyxuan0311/cuOP.git
mkdir build && cd build
cmake ..
make -j
```
编译好后会在项目根下的**lib**目录下生成对应的动态库文件：
- `libcuop_cublas.so` - cuBLAS算子库
- `libcuop_cudnn.so` - cuDNN算子库  
- `libcuop_jit.so` - JIT实时编译库
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
```

### JIT系统使用示例
```cpp
#include "jit/jit_wrapper.hpp"
#include "cuda_op/detail/cuBlas/gemm.hpp"

// 创建原始算子
Gemm<float> gemm;
gemm.SetWeight(weight);

// 创建JIT包装器（一行代码启用JIT优化）
JITWrapper<Gemm<float>> jit_gemm(gemm);
jit_gemm.EnableJIT(true);

// 使用方式与原始算子完全相同
jit_gemm.Forward(input, output);
```

## 常见问题
- **no CUDA-capable device is detected**
  - 请确保已正确安装 CUDA 驱动，并且 `ldd ./bench_gemm | grep cuda` 能看到 libcudart.so 路径。
- **链接错误**
  - 请检查 CMakeLists.txt 是否正确链接了 cudart、cublas、cudnn、nvrtc、glog 等依赖。
- **JIT编译失败**
  - 检查NVRTC是否正确安装，确保CUDA版本支持JIT编译。
  - 查看日志输出，JIT系统会自动回退到原始算子执行。

## 核心特性

### 🚀 **内存池优化**
- **多级缓存**: 线程本地、全局、预分配缓存
- **智能管理**: 自动内存碎片整理和过期缓存清理
- **性能提升**: 显著减少内存分配开销

### 🔧 **JIT实时编译**
- **零侵入性**: 包装器模式，现有代码完全不变
- **智能优化**: 多级内核架构，自动选择最优配置
- **硬件感知**: 充分利用Tensor Core、TMA等最新特性

### 📊 **算子支持**
- **cuBLAS**: GEMM、GEMV等高性能BLAS算子
- **cuDNN**: Conv、Pool、ReLU等深度学习算子
- **扩展性**: 插件化架构，易于添加新算子

## 贡献
欢迎提交 issue 和 PR，建议先阅读代码结构和注释。详细文档请参考 `src/jit/jit_docs.md`。

## License

MIT License
