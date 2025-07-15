# GlobalMaxPool2D 算子介绍

`GlobalMaxPool2D` 算子用于对一个二维输入张量（通常是图像的特征图）执行全局最大池化操作。它会计算整个输入张量的最大值，并输出一个包含单个值的标量张量。

## 实现原理

该算子的核心是 `globalmaxpool2D_kernel` CUDA kernel。这个 kernel 使用了并行归约（parallel reduction）的算法来高效地在 GPU 上找到最大值。

### 并行归约

并行归约是一种在并行处理器上将一组值合并为单个值的常用技术。在我们的场景中，我们希望找到输入张量中的最大值。

该算法主要包含以下步骤：

1.  **分块处理 (Grid-Stride Loop)**: 输入张量被逻辑上划分为多个块，每个 CUDA block 负责处理一部分数据。为了处理任意大小的输入，kernel 使用了 grid-stride 循环。每个线程会计算自己所分配到数据中的局部最大值。

2.  **共享内存中的块内归约 (Intra-Block Reduction)**: 在每个 block 内部，线程们将它们的局部最大值写入共享内存。然后，它们协作进行树状的归约操作，以找出该 block 内所有线程计算出的局部最大值中的最大值。这个过程是对数的复杂度（`log(blockSize)`），非常高效。

3.  **多级归约 (Multi-Pass Reduction)**: 如果输入张量非常大，以至于需要多个 CUDA block 来处理，那么一次 kernel 调用会产生多个部分最大值（每个 block 一个）。在这种情况下，会启动第二次 kernel 调用，对第一次调用产生的部分最大值再次进行归约，从而得到最终的全局最大值。

### Kernel (`globalmaxpool2D_kernel`) 详解

-   **输入**: `input` (输入张量), `output` (输出张量), `n` (输入张量元素总数)。
-   **共享内存**: `extern __shared__ T sdata[]` 用于在 block 内进行高效的归约操作。
-   **Grid-Stride 循环**: `while (i < n)` 循环确保所有元素都被处理，即使元素数量远大于线程总数。
-   **块内归约**: `for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)` 这个循环实现了树状的归约。在每次迭代中，一半的活动线程会将两个值进行比较和合并，从而使活动线程数减半。
-   **输出**: 每个 block 的 0 号线程 (`if (tid == 0)`) 会将该 block 的归约结果（部分最大值）写入全局内存。

## `Forward` 函数

`Forward` 函数是算子的入口。它负责：

1.  检查输入张量的维度是否正确。
2.  计算启动 kernel 所需的线程块（block）和线程（thread）数量。
3.  根据输入大小，决定是执行单次还是两次 kernel 调用来完成全局归约。
4.  调用 `globalmaxpool2D_kernel`。
5.  执行必要的 CUDA 错误检查和设备同步。

通过这种设计，`GlobalMaxPool2D` 算子能够高效、可扩展地处理不同大小的输入张量。