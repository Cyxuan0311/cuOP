#include "cuda_op/detail/cuDNN/batchnorm.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <cmath>

namespace cu_op_mem {

// Warp级别归约函数
__device__ __forceinline__ void warpReduceSum(float& val) {
    const int warpSize = 32; // 硬编码warp大小
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
}

__device__ __forceinline__ void warpReduceSum(double& val) {
    const int warpSize = 32; // 硬编码warp大小
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
}

// 为float类型创建优化的kernel函数
__global__ void batchnorm_forward_optimized_kernel_float(const float* input, float* output,
                                                         const float* gamma, const float* beta,
                                                         float* running_mean, float* running_var,
                                                         int N, int C, int HxW, float eps) {
    int c = blockIdx.x;
    int tid = threadIdx.x;
    const int warpSize = 32;
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    int num = N * HxW;
    
    // 使用更少的共享内存，只存储warp级别的结果
    extern __shared__ float batchnorm_shared_mem_float[];
    float* warp_sums = batchnorm_shared_mem_float;
    float* warp_sqsums = batchnorm_shared_mem_float + (blockDim.x + warpSize - 1) / warpSize;

    float sum = 0, sqsum = 0;
    
    // 向量化内存访问
    for (int i = tid; i < num; i += blockDim.x) {
        int n = i / HxW;
        int hw = i % HxW;
        float v = input[(n * C + c) * HxW + hw];
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
    
    float mean = sum / num;
    float var = sqsum / num - mean * mean;
    
    if (tid == 0) {
        running_mean[c] = mean;
        running_var[c] = var;
    }
    __syncthreads();
    
    // 使用rsqrtf提高性能
    float inv_std = rsqrtf(var + eps);
    
    // 向量化输出计算
    for (int i = tid; i < num; i += blockDim.x) {
        int n = i / HxW;
        int hw = i % HxW;
        int idx = (n * C + c) * HxW + hw;
        float v = input[idx];
        float norm = (v - mean) * inv_std;
        output[idx] = norm * gamma[c] + beta[c];
    }
}

// 为double类型创建优化的kernel函数
__global__ void batchnorm_forward_optimized_kernel_double(const double* input, double* output,
                                                          const double* gamma, const double* beta,
                                                          double* running_mean, double* running_var,
                                                          int N, int C, int HxW, double eps) {
    int c = blockIdx.x;
    int tid = threadIdx.x;
    const int warpSize = 32;
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    int num = N * HxW;
    
    // 使用更少的共享内存，只存储warp级别的结果
    extern __shared__ double batchnorm_shared_mem_double[];
    double* warp_sums = batchnorm_shared_mem_double;
    double* warp_sqsums = batchnorm_shared_mem_double + (blockDim.x + warpSize - 1) / warpSize;

    double sum = 0, sqsum = 0;
    
    // 向量化内存访问
    for (int i = tid; i < num; i += blockDim.x) {
        int n = i / HxW;
        int hw = i % HxW;
        double v = input[(n * C + c) * HxW + hw];
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
    
    double mean = sum / num;
    double var = sqsum / num - mean * mean;
    
    if (tid == 0) {
        running_mean[c] = mean;
        running_var[c] = var;
    }
    __syncthreads();
    
    // 使用rsqrt提高性能
    double inv_std = 1.0 / sqrt(var + eps);
    
    // 向量化输出计算
    for (int i = tid; i < num; i += blockDim.x) {
        int n = i / HxW;
        int hw = i % HxW;
        int idx = (n * C + c) * HxW + hw;
        double v = input[idx];
        double norm = (v - mean) * inv_std;
        output[idx] = norm * gamma[c] + beta[c];
    }
}

template <typename T>
StatusCode BatchNorm<T>::Forward(const Tensor<T>& input, Tensor<T>& output,
                                 const Tensor<T>& gamma, const Tensor<T>& beta,
                                 Tensor<T>& running_mean, Tensor<T>& running_var,
                                 T eps) {
    const auto& shape = input.shape();
    int N, C, HxW;
    if (shape.size() == 4) {
        N = shape[0];
        C = shape[1];
        HxW = shape[2] * shape[3];
    } else if (shape.size() == 2) {
        N = shape[0];
        C = shape[1];
        HxW = 1;
    } else {
        LOG(ERROR) << "BatchNorm only supports 2D [N,C] or 4D [N,C,H,W] input.";
        return StatusCode::SHAPE_MISMATCH;
    }
    if (output.data() == nullptr || output.shape() != shape) {
        output = Tensor<T>(shape);
    }
    if (running_mean.data() == nullptr || running_mean.shape() != std::vector<std::size_t>{(size_t)C}) {
        running_mean = Tensor<T>({(size_t)C});
    }
    if (running_var.data() == nullptr || running_var.shape() != std::vector<std::size_t>{(size_t)C}) {
        running_var = Tensor<T>({(size_t)C});
    }
    int threads = 256;
    int blocks = C;
    // 减少共享内存使用，只存储warp级别的结果
    const int warpSize = 32;
    size_t shared_mem = ((threads + warpSize - 1) / warpSize) * 2 * sizeof(T);
    
    // 根据类型调用相应的优化kernel函数
    if constexpr (std::is_same_v<T, float>) {
        batchnorm_forward_optimized_kernel_float<<<blocks, threads, shared_mem>>>(
            input.data(), output.data(),
            gamma.data(), beta.data(), running_mean.data(), running_var.data(),
            N, C, HxW, eps);
    } else if constexpr (std::is_same_v<T, double>) {
        batchnorm_forward_optimized_kernel_double<<<blocks, threads, shared_mem>>>(
            input.data(), output.data(),
            gamma.data(), beta.data(), running_mean.data(), running_var.data(),
            N, C, HxW, eps);
    } else {
        LOG(ERROR) << "BatchNorm only supports float and double types.";
        return StatusCode::UNSUPPORTED_OPERATION;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "BatchNorm kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    cudaDeviceSynchronize();
    return StatusCode::SUCCESS;
}

template class BatchNorm<float>;
template class BatchNorm<double>;

} // namespace cu_op_mem
