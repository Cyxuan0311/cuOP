# CUDA Softmax 算子实现详解


## 概述
本文档详细介绍基于CUDA实现的Softmax算子，重点解析核心计算内核（`softmax_kernel`）的设计思路、并行策略及优化细节。该实现通过GPU并行计算加速Softmax运算，兼顾数值稳定性与计算效率，适用于深度学习中的分类任务输出层或注意力机制等场景。


## 1. Softmax 算法原理
Softmax用于将一组数值映射为概率分布（总和为1），公式如下：  
对于输入向量 \( x = [x_1, x_2, ..., x_n] \)，输出为：  
\[ \text{Softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^n e^{x_j - \max(x)}} \]  

**关键特性**：  
- 减去输入向量的最大值（\(\max(x)\)）以避免指数运算溢出（数值稳定性优化）；  
- 输出值均在 \((0, 1)\) 范围内，且总和为1，满足概率分布特性。  


## 2. 核心内核（`softmax_kernel`）解析
`softmax_kernel` 是算子的核心计算函数，负责在GPU上并行执行Softmax运算。其设计围绕“按行并行处理”和“高效归约计算”展开，充分利用CUDA的线程层次结构（线程、Warp、线程块）。


### 2.1 函数签名与参数说明
```cpp
template <typename T>
__global__ void softmax_kernel(
    const T* input,    // 输入张量（device端指针，shape: [rows, cols]）
    T* output,         // 输出张量（device端指针，shape与输入一致）
    std::size_t rows,  // 输入行数（样本数或特征数）
    std::size_t cols   // 输入列数（每个样本的特征维度）
);
```


### 2.2 并行策略设计
- **线程块与线程分工**：  
  - 每个线程块（`block`）负责处理输入中的一行（`row`），通过 `blockIdx.x` 定位行索引；  
  - 线程块内的线程（`threadIdx.x`）按列分工，每个线程处理多个列（通过 `col += blockDim.x` 循环），覆盖整行所有元素。  


### 2.3 内核执行流程
内核执行分为3个核心步骤，均通过“线程级并行+归约”实现高效计算：

#### 步骤1：计算当前行的最大值（数值稳定性基础）
1. **线程局部最大值计算**：每个线程遍历负责的列，记录局部最大值：  
   ```cpp
   T thread_max = -INFINITY;
   for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
       thread_max = max(thread_max, input[row * cols + col]);
   }
   ```  
2. **块级归约求全局最大值**：通过 `blockReduceMax` 函数将线程局部最大值归约为行全局最大值，并存储到共享内存 `row_max` 中：  
   ```cpp
   thread_max = blockReduceMax(thread_max);  // 块内所有线程的最大值归约
   if (threadIdx.x == 0) row_max = thread_max;  // 共享内存存储行最大值
   __syncthreads();  // 确保所有线程可见行最大值
   ```  


#### 步骤2：计算指数值并累加总和
1. **线程局部指数与求和**：每个线程计算负责列的指数值（减去行最大值避免溢出），并累加局部总和：  
   ```cpp
   T thread_sum = 0;
   for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
       T exp_val = exp(input[row * cols + col] - row_max);  // 减去最大值
       output[row * cols + col] = exp_val;  // 暂存指数值
       thread_sum += exp_val;  // 累加局部总和
   }
   ```  
2. **块级归约求全局总和**：通过 `blockReduceSum` 函数将线程局部总和归约为行全局总和，存储到共享内存 `row_sum` 中：  
   ```cpp
   thread_sum = blockReduceSum(thread_sum);  // 块内所有线程的总和归约
   if (threadIdx.x == 0) row_sum = thread_sum;  // 共享内存存储行总和
   __syncthreads();  // 确保所有线程可见行总和
   ```  


#### 步骤3：归一化计算（得到最终Softmax结果）
每个线程对负责的列执行归一化，将指数值除以行总和：  
```cpp
for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
    output[row * cols + col] /= row_sum;  // 归一化得到概率分布
}
```  


### 2.4 归约函数（`warpReduce`/`blockReduce`）解析
归约（Reduce）是并行计算中聚合局部结果的核心操作。本实现通过“Warp级归约”+“Block级归约”两级结构，高效计算行最大值和总和。

#### 2.4.1 Warp级归约（`warpReduceMax`/`warpReduceSum`）
Warp是GPU的基本执行单元（NVIDIA GPU中一个Warp含32个线程），通过 `__shfl_down_sync` 指令实现Warp内线程通信，无共享内存开销：  
```cpp
//  warp内最大值归约
template <typename T>
__device__ T warpReduceMax(T val) {
    for(int offset = (warpSize / 2); offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));  // 线程间数据交换
    return val;
}

//  warp内总和归约（逻辑类似，将max替换为加法）
template <typename T>
__device__ T warpReduceSum(T val) {
    for(int offset = (warpSize / 2); offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```  
**原理**：通过逐步减半偏移量（16→8→4→...→1），将Warp内所有线程的数值聚合到第一个线程。


#### 2.4.2 Block级归约（`blockReduceMax`/`blockReduceSum`）
当线程块大小超过Warp尺寸（32）时，需先通过Warp级归约得到每个Warp的结果，再通过共享内存聚合为块级结果：  
```cpp
template <typename T>
__device__ T blockReduceMax(T val) {
    static __shared__ T shared[32];  // 存储每个Warp的归约结果（最多32个Warp）
    int lane = threadIdx.x % warpSize;  // 线程在Warp内的索引
    int wid = threadIdx.x / warpSize;   // Warp索引

    val = warpReduceMax(val);  // 先做Warp级归约

    if(lane == 0) shared[wid] = val;  // 每个Warp的第一个线程存储结果到共享内存
    __syncthreads();

    // 仅用第一个Warp处理共享内存中的Warp结果
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -INFINITY;
    if(wid == 0) val = warpReduceMax(val);  // 最终Block级归约

    return val;
}
```  
**优势**：减少共享内存占用（仅需32个元素），同时利用Warp级指令的高效性。


## 3. 前向传播接口（`Softmax<T>::Forward`）
`Forward` 方法是算子的对外接口，负责参数校验、内核配置与执行控制。

### 3.1 核心流程
1. **输入校验**：检查输入是否为二维张量（`[rows, cols]`），若维度不匹配则返回错误。  
2. **输出初始化**：确保输出张量的形状与输入一致，若未初始化则创建新张量。  
3. **内核配置**：  
   - 线程块大小（`threads_per_block`）：256（兼顾并行度与资源利用率）；  
   - 网格大小（`blocks_per_grid`）：等于行数（`rows`），每个线程块处理一行。  
4. **内核执行与错误处理**：  
   - 启动 `softmax_kernel` 并同步设备；  
   - 通过 `cudaGetLastError()` 检查内核启动错误，确保运算正确性。  


## 4. 优化亮点与特性
1. **数值稳定性保障**：通过减去每行最大值避免指数运算溢出（解决 \( e^x \) 在 \( x \) 较大时的数值爆炸问题）。  
2. **高效归约设计**：  
   - 基于Warp的 `__shfl_down_sync` 指令实现低延迟线程通信；  
   - 两级归约（Warp→Block）平衡并行度与内存开销。  
3. **并行策略合理**：  
   - 按行并行（每行一个线程块），符合Softmax“行内独立计算”的特性；  
   - 线程按列分工（每个线程处理多个列），充分利用线程资源。  
4. **模板化设计**：支持 `float` 和 `double` 数据类型，适配不同精度需求。  
5. **共享内存高效利用**：仅存储每行的最大值和总和（2个元素），避免内存浪费。  


## 5. 使用注意事项
1. **输入维度限制**：当前实现仅支持二维张量（`[rows, cols]`）。若需处理更高维度（如4D张量 `[batch, channel, height, width]`），需扩展内核以支持批量或通道维度（通常在最后一个维度上执行Softmax）。  
2. **线程块大小调优**：默认线程块大小为256，可根据GPU架构（如Ampere、Hopper）调整（建议为32的倍数，匹配Warp尺寸）。  
3. **性能考量**：  
   - 当 `cols` 较小时（如小于256），可减少线程块大小以避免线程空闲；  
   - 对于超大 `cols`，可通过循环分块处理（当前已支持，通过 `col += blockDim.x` 实现）。  
4. **同步开销**：`cudaDeviceSynchronize()` 会阻塞主机端执行，实际部署中可结合流水线设计（如异步内存拷贝+内核执行）优化。  


## 6. 扩展方向
- 支持高维张量（如批量维度 `[batch, cols]` 或图像维度 `[batch, channel, H, W]`）；  
- 增加对对数Softmax（LogSoftmax）的支持（直接计算 `log(Softmax(x))`，减少数值精度损失）；  
- 结合CUDA Tensor Core加速半精度（`half`）计算，提升吞吐量；  
- 支持沿指定维度执行Softmax（通过参数 `dim` 配置），增强通用性。  


## 总结
本实现基于CUDA的并行计算模型，通过高效归约策略和数值稳定性优化，实现了高性能的Softmax算子。核心内核 `softmax_kernel` 充分利用Warp级通信与两级归约，在保证数值正确性的同时，最大化GPU的并行计算能力，适用于深度学习中的分类、注意力机制等场景。

## 0. 多维/四维张量与 dim 支持

- 现已支持 shape.size() >= 2 的多维张量（如 [N, C, H, W]），可在任意维度（如 channel/width）上做 softmax。
- 新增 dim 参数，指定在哪一维做 softmax，兼容 PyTorch 的 `dim` 用法。
- 自动展平其他维度为 batch，softmax 仅在目标维度做。
- 原二维接口完全兼容。

### 用法示例（四维张量）
```cpp
Tensor<float> input({2, 3, 4, 5}); // 4D 输入
Tensor<float> output;
Softmax<float> softmax;
softmax.Forward(input, output, 1); // 在第1维（C）做softmax
```

### 注意事项
- 支持 shape.size() >= 2 的任意张量。
- dim 支持负数（如 -1 表示最后一维）。
- 推荐用法与 PyTorch 保持一致。