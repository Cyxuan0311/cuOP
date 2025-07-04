# cuOP

本项目为基于 **CUDA** 的算子与内存管理库。其中使用了内存池进行了对算子的分配的优化，同时针对不同的**HPC**应用方向有相应的算子的**CUDA**实现

## 目录结构

```
include/   项目头文件目录，包含所有对外接口声明。
  - cuda_op：CUDA 相关算子与抽象接口头文件。
  - detail：相关算子头文件，如 GEMM。
  - data：张量（Tensor）、内存池等数据结构相关头文件。

src/       项目源代码实现目录。
  - base：基础设施实现，如内存池。
  - cuda_op：CUDA 相关算子实现。


bench/   性能测试与基准测试代码。
  - cuBlas：Blas算子性能测试代码及其 CMake 构建脚本。

test/    单元测试代码。

.clang-format：代码风格配置文件。

CMakeLists.txt：CMake 构建脚本。

README.md：项目说明文档。
```

## 主要功能

- **Tensor 管理**：支持多维张量的高效分配与释放，基于自定义 CUDA 内存池。
  
- **算子**：基于 cuBLAS 封装的高性能矩阵乘法。
```
· cuDNN 深度学习相关算子
· cuBLAS 基础线性代数子程序（BLAS）算子
```
- **内存池**：减少频繁 cudaMalloc/cudaFree 带来的性能损耗。

- **性能基准**：性能测试。

## 编译与运行

### 依赖

- CUDA 11.5 及以上（推荐 CUDA 12.x）
- cuBLAS
- glog
- CMake 3.14 及以上
- GCC 7.0 及以上
- google benchmark

### 常见问题
- no CUDA-capable device is detected
请确保已正确安装 CUDA 驱动，并且 ldd ./bench_gemm | grep cuda 能看到 libcudart.so 路径。
- 链接错误
请检查 CMakeLists.txt 是否正确链接了 cudart、cublas 和 glog。

## 贡献
欢迎提交 issue 和 PR，建议先阅读代码结构和注释。

## License

MIT License