#include "cuda_op/detail/cuDNN/softmax.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>

namespace cu_op_mem {

template <typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
    for(int offset = (warpSize / 2); offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    for(int offset = (warpSize / 2); offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// 数值稳定的exp函数
template <typename T>
__device__ __forceinline__ T stable_exp(T x) {
    const T max_val = T(88.0);  // 避免溢出
    const T min_val = T(-88.0); // 避免下溢
    x = fmaxf(fminf(x, max_val), min_val);
    return exp(x);
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
    
    // 2. 计算指数值并累加总和（使用数值稳定的exp）
    T thread_sum = 0;
    for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        T exp_val = stable_exp(input[row * cols + col] - row_max);
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

// 新的 kernel，支持 batch 维度和任意 softmax 维度（dim）
template <typename T>
__global__ void softmax_nd_kernel(const T* input, T* output, std::size_t batch, std::size_t dim_size, std::size_t inner_stride, std::size_t outer_stride) {
    // 每个block处理一个softmax向量
    std::size_t idx = blockIdx.x;
    if (idx >= batch) return;
    const T* in_ptr = input + idx * inner_stride;
    T* out_ptr = output + idx * inner_stride;
    // 1. 找最大值
    T max_val = -INFINITY;
    for (std::size_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        T v = in_ptr[i * outer_stride];
        max_val = max(max_val, v);
    }
    max_val = blockReduceMax(max_val);
    __shared__ T row_max;
    if (threadIdx.x == 0) row_max = max_val;
    __syncthreads();
    // 2. 计算exp和sum（使用数值稳定的exp）
    T sum = 0;
    for (std::size_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        T v = stable_exp(in_ptr[i * outer_stride] - row_max);
        out_ptr[i * outer_stride] = v;
        sum += v;
    }
    sum = blockReduceSum(sum);
    __shared__ T row_sum;
    if (threadIdx.x == 0) row_sum = sum;
    __syncthreads();
    // 3. 归一化
    for (std::size_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        out_ptr[i * outer_stride] /= row_sum;
    }
}

// 混合精度softmax kernel（使用half精度进行中间计算）
__global__ void softmax_mixed_precision_kernel(const float* input, float* output, 
                                              std::size_t rows, std::size_t cols) {
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;
    
    __shared__ float row_max;
    __shared__ float row_sum;
    
    // 1. 计算最大值（使用float精度）
    float thread_max = -INFINITY;
    for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        thread_max = max(thread_max, input[row * cols + col]);
    }
    
    // Warp级别归约
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    
    if (threadIdx.x == 0) {
        row_max = thread_max;
    }
    __syncthreads();
    
    // 2. 计算exp和sum（使用数值稳定的exp）
    float thread_sum = 0;
    for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        float val = input[row * cols + col] - row_max;
        float exp_val = stable_exp(val);
        
        output[row * cols + col] = exp_val;
        thread_sum += exp_val;
    }
    
    // Warp级别归约
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
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
StatusCode Softmax<T>::Forward(const Tensor<T>& input, Tensor<T>& output, int dim) {
    std::vector<std::size_t> shape = input.shape();
    if (shape.size() < 2) {
        LOG(ERROR) << "Softmax requires at least 2D tensor, got " << shape.size() << "D";
        return StatusCode::SHAPE_MISMATCH;
    }
    if (dim < 0) dim += shape.size();
    if (dim < 0 || dim >= (int)shape.size()) {
        LOG(ERROR) << "Softmax dim out of range: " << dim;
        return StatusCode::SHAPE_MISMATCH;
    }
    // 计算 batch, dim_size, inner_stride, outer_stride
    std::size_t batch = 1, dim_size = shape[dim], inner_stride = 1, outer_stride = 1;
    for (int i = 0; i < dim; ++i) batch *= shape[i];
    for (int i = dim + 1; i < (int)shape.size(); ++i) inner_stride *= shape[i];
    outer_stride = inner_stride;
    batch *= inner_stride;
    // 输出 shape 与输入一致
    if (output.data() == nullptr || output.shape() != shape) {
        output = Tensor<T>(shape);
    }
    // 支持 float/double 和混合精度
    const std::size_t threads_per_block = 256;
    const std::size_t blocks_per_grid = batch;
    
    // 2D特例，根据类型选择kernel
    if (shape.size() == 2 && dim == 1) {
        if constexpr (std::is_same_v<T, float>) {
            // 对于float类型，可以选择使用混合精度kernel
            softmax_mixed_precision_kernel<<<blocks_per_grid, threads_per_block>>>(
                input.data(), output.data(), shape[0], shape[1]);
        } else {
            softmax_kernel<T><<<blocks_per_grid, threads_per_block>>>(
                input.data(), output.data(), shape[0], shape[1]);
        }
    } else {
        softmax_nd_kernel<T><<<blocks_per_grid, threads_per_block>>>(
            input.data(), output.data(), batch, dim_size, inner_stride, outer_stride);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "Softmax kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    cudaDeviceSynchronize();
    return StatusCode::SUCCESS;
}

// 显式实例化模板类
template class Softmax<float>;
template class Softmax<double>;

} // namespace cu_op_mem