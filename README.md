# cuOP

![C++17](https://img.shields.io/badge/C++-17-blue)
![CUDA](https://img.shields.io/badge/CUDA-11.5%2B-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![CMake](https://img.shields.io/badge/CMake-3.20.x-red)
![Repo Size](https://img.shields.io/github/repo-size/Cyxuan0311/cuOP)

高性能 CUDA 算子与内存管理库，支持高效的张量运算、内存池优化和 HPC/深度学习常用算子。

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
│   └── data/               # 张量（Tensor）等数据结构
├── src/                    # 源码实现
│   ├── base/               # 内存池等实现
│   └── cuda_op/            # CUDA 算子实现
│       └── detail/         # 具体算子实现
├── bench/                  # 性能基准测试
│   └── cuBlas/             # BLAS 算子基准
│       ├── gemm/           # GEMM 性能测试
│       └── gemv/           # GEMV 性能测试
├── test/                   # 单元测试
│   └── cuBlas/             # BLAS 算子测试
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
- **内存池优化**：减少频繁 cudaMalloc/cudaFree 带来的性能损耗。
- **性能基准**：提供标准化的算子性能测试。

## 编译与运行

### 依赖
- CUDA 11.5 及以上（推荐 CUDA 12.x）
- cuBLAS（如需 BLAS 算子）
- cuDNN（如需深度学习算子）
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
编译好后会在项目根下的**lib**目录下生成对应的动态库文件，比如会有Blas库或DNN库。
### 运行基准测试
```bash
# 示例
./bench/cuBlas/gemm/bench_gemm
./bench/cuBlas/gemv/bench_gemv
....
```

### 运行单元测试
```bash
# 示例
./test/cuBlas/test_gemm/test_gemm
./test/cuBlas/test_gemv/test_gemv
....
```

## 常见问题
- **no CUDA-capable device is detected**
  - 请确保已正确安装 CUDA 驱动，并且 `ldd ./bench_gemm | grep cuda` 能看到 libcudart.so 路径。
- **链接错误**
  - 请检查 CMakeLists.txt 是否正确链接了 cudart、cublas、glog 等依赖。

## 贡献
欢迎提交 issue 和 PR，建议先阅读代码结构和注释。

## License

MIT License
