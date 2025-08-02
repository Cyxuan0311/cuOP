# GlobalAveragePool2D 简介

`GlobalAveragePool2D` 是一个基于 CUDA 的全局平均池化（Global Average Pooling）算子，适用于二维张量的全局特征提取。

## 功能说明
- 对输入的二维张量（Tensor）所有元素求平均，输出为单一标量。
- 常用于卷积神经网络的全局特征聚合。

## CUDA Kernel 实现与优化
### 实现原理
- kernel 启动一个 block，block 内多个线程协作完成所有输入元素的累加。
- 每个线程负责累加部分输入元素（分片遍历），将结果写入共享内存。
- 线程块内通过归约（reduction）操作，将所有线程的部分和累加为总和。
- 由 thread 0 计算平均值并写入输出。

### 优化点
- **共享内存归约**：
  - 利用 shared memory 存储每个线程的部分和，极大减少全局内存访问和原子操作。
  - 采用分步归约（stride 递减），高效聚合所有线程的结果。
- **并行累加**：
  - 每个线程处理多个输入元素，充分利用线程并行性。
- **适用场景**：
  - 适合输入为单个二维张量（如特征图、图像）的全局平均池化。

## 接口说明
- 类名：`GlobalAveragePool2D<T>`
- 构造函数：
  ```cpp
  GlobalAveragePool2D();
  ```
- 前向计算：
  ```cpp
  StatusCode Forward(const Tensor<T>& input, Tensor<T>& output);
  ```
  - `input`：输入二维张量（H x W）
  - `output`：输出单元素张量

## 用法示例
```cpp
#include "cuda_op/detail/cuDNN/globalaverpool2D.hpp"
using namespace cu_op_mem;

Tensor<float> input({32, 32}); // 32x32 输入
Tensor<float> output;
GlobalAveragePool2D<float> gap;
gap.Forward(input, output);
// output[0] 即为全局平均值
```

## 四维张量支持
- 现已支持 shape.size() == 4 的输入（如 [N, C, H, W]），对每个 [N, C] 独立做全局池化，输出 shape 为 [N, C, 1]。
- 原二维接口完全兼容。

### 四维用法示例
```cpp
Tensor<float> input({2, 3, 32, 32}); // 4D 输入
Tensor<float> output;
GlobalAveragePool2D<float> gap;
gap.Forward(input, output); // 输出 shape 为 [2, 3, 1]
```

### 注意事项
- 支持 2D/4D 张量输入。

## 注意事项
- 输入必须为二维张量。
- 输出为单元素张量（标量）。

---
如需进一步优化或定制功能，请联系开发者。
