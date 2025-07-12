#include "cuda_op/detail/cuDNN/softmax.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>

namespace cu_op_mem {

template <typename T>
__device__ T warpReduceMax(T val) {
    for(int offset = (warpSize / 2); offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

template <typename T>
__device__ T warpReduceSum(T val) {
    for(int offset = (warpSize / 2); offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <typename T>
__device__ T blockReduceMax(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceMax(val);

    if(lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -INFINITY;
    if(wid == 0) val = warpReduceMax(val);

    return val;
}

template <typename T>
__device__ T blockReduceSum(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if(lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if(wid == 0) val = warpReduceSum(val);

    return val;
}

template <typename T>
__global__ void softmax_kernel(const T* input, T* output, std::size_t rows, std::size_t cols) {
    // 每个块处理一行数据
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;
    
    // 共享内存用于存储当前行的最大值和总和
    __shared__ T row_max;
    __shared__ T row_sum;
    
    // 1. 计算当前行的最大值
    T thread_max = -INFINITY;
    for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        thread_max = max(thread_max, input[row * cols + col]);
    }
    
    thread_max = blockReduceMax(thread_max);
    
    if (threadIdx.x == 0) {
        row_max = thread_max;
    }
    __syncthreads();
    
    // 2. 计算指数值并累加总和
    T thread_sum = 0;
    for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        T exp_val = exp(input[row * cols + col] - row_max);
        output[row * cols + col] = exp_val;
        thread_sum += exp_val;
    }
    
    thread_sum = blockReduceSum(thread_sum);
    
    if (threadIdx.x == 0) {
        row_sum = thread_sum;
    }
    __syncthreads();
    
    // 3. 归一化
    for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        output[row * cols + col] /= row_sum;
    }
}

template <typename T>
StatusCode Softmax<T>::Forward(const Tensor<T>& input, Tensor<T>& output) {
    // 获取输入张量的维度信息
    std::vector<std::size_t> shape = input.shape();
    
    // 检查输入是否为二维张量 [rows, cols]
    if (shape.size() != 2) {
        LOG(ERROR) << "Softmax requires 2D tensor, but got " << shape.size() << "D tensor";
        return StatusCode::SHAPE_MISMATCH;
    }
    
    std::size_t rows = shape[0];
    std::size_t cols = shape[1];
    
    // 确保输出张量已初始化且大小正确
    if (output.data() == nullptr || output.shape() != shape) {
        output = Tensor<T>(shape);
    }
    
    // 配置核函数执行参数
    const std::size_t threads_per_block = 256;
    const std::size_t blocks_per_grid = rows;
    
    LOG(INFO) << "Softmax launch: blocks = " << blocks_per_grid 
              << " , threads = " << threads_per_block 
              << " , rows = " << rows 
              << " , cols = " << cols;
    
    // 执行核函数
    softmax_kernel<T><<<blocks_per_grid, threads_per_block>>>(
        input.data(), output.data(), rows, cols);
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "Softmax kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    
    // 同步设备以确保所有操作完成
    cudaDeviceSynchronize();
    
    return StatusCode::SUCCESS;
}

// 显式实例化模板类
template class Softmax<float>;
template class Softmax<double>;

} // namespace cu_op_mem