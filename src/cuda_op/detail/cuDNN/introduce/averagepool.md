# AveragePool2D 简介

`AveragePool2D` 是一个基于 CUDA 的二维平均池化（Average Pooling）算子，支持高效的 GPU 加速。

## 功能说明
- 对输入的二维张量（Tensor）进行平均池化操作。
- 支持自定义池化窗口大小（pool_height, pool_width）和步幅（stride_height, stride_width）。
- 适用于图像、特征图等二维数据的降采样。

## CUDA Kernel 实现与优化
### 实现原理
- 每个线程负责计算输出张量中的一个元素（即一个池化窗口的平均值）。
- 线程根据输出坐标，计算其对应的输入窗口区域。
- 对窗口内所有元素求和并计数，最后取平均值写入输出。

### 优化点
- **共享内存（Shared Memory）优化**：
  - 使用 tile/block 方式将输入数据块加载到共享内存，减少全局内存访问次数。
  - 线程块内协作加载和处理数据，提高内存访问效率。
- **边界处理**：
  - 对于池化窗口超出输入边界的情况，自动裁剪，保证不会越界访问。
- **并行化**：
  - 充分利用 CUDA 的线程并行能力，适合大规模二维数据的高效池化。
- **适用场景**：
  - 该实现对大窗口、步幅较小（窗口重叠多）的场景尤为高效。

## 接口说明
- 类名：`AveragePool2D<T>`
- 构造函数：
  ```cpp
  AveragePool2D(int pool_height, int pool_width, int stride_height = 1, int stride_width = 1);
  ```
- 前向计算：
  ```cpp
  StatusCode Forward(const Tensor<T>& input, Tensor<T>& output);
  ```
  - `input`：输入二维张量（H x W）
  - `output`：输出池化结果张量

## 用法示例
```cpp
#include "cuda_op/detail/cuDNN/averagepool2D.hpp"
using namespace cu_op_mem;

Tensor<float> input({32, 32}); // 32x32 输入
Tensor<float> output;
AveragePool2D<float> avgpool(2, 2, 2, 2); // 2x2窗口，步幅2
avgpool.Forward(input, output);
```

## 四维张量支持
- 现已支持 shape.size() == 4 的输入（如 [N, C, H, W]），对每个 [N, C] 独立做池化。
- 原二维接口完全兼容。

### 四维用法示例
```cpp
Tensor<float> input({2, 3, 32, 32}); // 4D 输入
Tensor<float> output;
AveragePool2D<float> avgpool(2, 2, 2, 2);
avgpool.Forward(input, output); // 自动对每个 [N, C] 做池化
```

## 性能优化说明
- 内核实现采用共享内存（tile）优化，减少全局内存访问。
- 线程块协作处理池化窗口，适合大窗口和有重叠的情况。
- 支持 float/double 类型。

## 注意事项
- 输入必须为二维张量。
- 池化窗口和步幅需为正整数，且不能超过输入尺寸。

---
如需进一步优化或定制功能，请联系开发者。
