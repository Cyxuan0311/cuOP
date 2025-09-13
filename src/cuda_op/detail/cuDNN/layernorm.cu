#include "cuda_op/detail/cuDNN/layernorm.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <cmath>

namespace cu_op_mem {

// Warp级别归约函数
__device__ __forceinline__ void warpReduceSum(float& val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
}

__device__ __forceinline__ void warpReduceSum(double& val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
}

// 优化的LayerNorm kernel for float
__global__ void layernorm_forward_optimized_kernel_float(const float* input, float* output,
                                                        const float* gamma, const float* beta,
                                                        int batch, int norm_size, int norm_stride, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    if (row >= batch) return;
    
    const float* in_ptr = input + row * norm_stride;
    float* out_ptr = output + row * norm_stride;
    
    // 使用更少的共享内存，只存储warp级别的结果
    extern __shared__ float layernorm_shared_mem_float[];
    float* warp_sums = layernorm_shared_mem_float;
    float* warp_sqsums = layernorm_shared_mem_float + (blockDim.x + warpSize - 1) / warpSize;

    // 1. 每个线程处理部分元素
    float sum = 0, sqsum = 0;
    for (int i = tid; i < norm_size; i += blockDim.x) {
        float v = in_ptr[i];
        sum += v;
        sqsum += v * v;
    }
    
    // Warp级别归约
    warpReduceSum(sum);
    warpReduceSum(sqsum);
    
    // 存储warp结果到共享内存
    if (lane == 0) {
        warp_sums[warp_id] = sum;
        warp_sqsums[warp_id] = sqsum;
    }
    __syncthreads();
    
    // 最终归约
    if (tid < (blockDim.x + warpSize - 1) / warpSize) {
        sum = warp_sums[tid];
        sqsum = warp_sqsums[tid];
    } else {
        sum = 0;
        sqsum = 0;
    }
    
    warpReduceSum(sum);
    warpReduceSum(sqsum);
    
    float mean = sum / norm_size;
    float var = sqsum / norm_size - mean * mean;
    
    // 使用rsqrtf提高性能
    float inv_std = rsqrtf(var + eps);
    
    // 3. 并行归一化写回
    for (int i = tid; i < norm_size; i += blockDim.x) {
        float v = in_ptr[i];
        float norm = (v - mean) * inv_std;
        out_ptr[i] = norm * gamma[i] + beta[i];
    }
}

// 优化的LayerNorm kernel for double
__global__ void layernorm_forward_optimized_kernel_double(const double* input, double* output,
                                                         const double* gamma, const double* beta,
                                                         int batch, int norm_size, int norm_stride, double eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    if (row >= batch) return;
    
    const double* in_ptr = input + row * norm_stride;
    double* out_ptr = output + row * norm_stride;
    
    // 使用更少的共享内存，只存储warp级别的结果
    extern __shared__ double layernorm_shared_mem_double[];
    double* warp_sums = layernorm_shared_mem_double;
    double* warp_sqsums = layernorm_shared_mem_double + (blockDim.x + warpSize - 1) / warpSize;

    // 1. 每个线程处理部分元素
    double sum = 0, sqsum = 0;
    for (int i = tid; i < norm_size; i += blockDim.x) {
        double v = in_ptr[i];
        sum += v;
        sqsum += v * v;
    }
    
    // Warp级别归约
    warpReduceSum(sum);
    warpReduceSum(sqsum);
    
    // 存储warp结果到共享内存
    if (lane == 0) {
        warp_sums[warp_id] = sum;
        warp_sqsums[warp_id] = sqsum;
    }
    __syncthreads();
    
    // 最终归约
    if (tid < (blockDim.x + warpSize - 1) / warpSize) {
        sum = warp_sums[tid];
        sqsum = warp_sqsums[tid];
    } else {
        sum = 0;
        sqsum = 0;
    }
    
    warpReduceSum(sum);
    warpReduceSum(sqsum);
    
    double mean = sum / norm_size;
    double var = sqsum / norm_size - mean * mean;
    
    // 使用rsqrt提高性能
    double inv_std = 1.0 / sqrt(var + eps);
    
    // 3. 并行归一化写回
    for (int i = tid; i < norm_size; i += blockDim.x) {
        double v = in_ptr[i];
        double norm = (v - mean) * inv_std;
        out_ptr[i] = norm * gamma[i] + beta[i];
    }
}

template <typename T>
StatusCode LayerNorm<T>::Forward(const Tensor<T>& input, Tensor<T>& output,
                                 const Tensor<T>& gamma, const Tensor<T>& beta,
                                 int normalized_dim, T eps) {
    const auto& shape = input.shape();
    int ndim = shape.size();
    if (normalized_dim < 0) normalized_dim += ndim;
    if (normalized_dim < 0 || normalized_dim >= ndim) {
        LOG(ERROR) << "LayerNorm: normalized_dim out of range.";
        return StatusCode::SHAPE_MISMATCH;
    }
    // 计算batch和norm_size
    int batch = 1, norm_size = 1;
    for (int i = 0; i < normalized_dim; ++i) batch *= shape[i];
    for (int i = normalized_dim; i < ndim; ++i) norm_size *= shape[i];
    int norm_stride = norm_size;
    if (output.data() == nullptr || output.shape() != shape) {
        output = Tensor<T>(shape);
    }
    int threads = 256;
    int blocks = batch;
    // 减少共享内存使用，只存储warp级别的结果
    size_t shared_mem = ((threads + warpSize - 1) / warpSize) * 2 * sizeof(T);
    
    // 根据类型调用相应的优化kernel函数
    if constexpr (std::is_same_v<T, float>) {
        layernorm_forward_optimized_kernel_float<<<blocks, threads, shared_mem>>>(
            input.data(), output.data(),
            gamma.data(), beta.data(), batch, norm_size, norm_stride, eps);
    } else if constexpr (std::is_same_v<T, double>) {
        layernorm_forward_optimized_kernel_double<<<blocks, threads, shared_mem>>>(
            input.data(), output.data(),
            gamma.data(), beta.data(), batch, norm_size, norm_stride, eps);
    } else {
        LOG(ERROR) << "LayerNorm only supports float and double types.";
        return StatusCode::UNSUPPORTED_OPERATION;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "LayerNorm kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    cudaDeviceSynchronize();
    return StatusCode::SUCCESS;
}

template class LayerNorm<float>;
template class LayerNorm<double>;

} // namespace cu_op_mem
