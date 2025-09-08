#include "cuda_op/detail/cuDNN/layernorm.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <cmath>

namespace cu_op_mem {

// 为float类型创建独立的kernel函数
__global__ void layernorm_forward_parallel_kernel_float(const float* input, float* output,
                                                        const float* gamma, const float* beta,
                                                        int batch, int norm_size, int norm_stride, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= batch) return;
    const float* in_ptr = input + row * norm_stride;
    float* out_ptr = output + row * norm_stride;
    extern __shared__ float layernorm_shared_mem_float[]; // layernorm_shared_mem_float[0:blockDim.x] for sum, [blockDim.x:2*blockDim.x] for sqsum

    // 1. 每个线程处理部分元素
    float sum = 0, sqsum = 0;
    for (int i = tid; i < norm_size; i += blockDim.x) {
        float v = in_ptr[i];
        sum += v;
        sqsum += v * v;
    }
    layernorm_shared_mem_float[tid] = sum;
    layernorm_shared_mem_float[blockDim.x + tid] = sqsum;
    __syncthreads();

    // 2. block内归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            layernorm_shared_mem_float[tid] += layernorm_shared_mem_float[tid + s];
            layernorm_shared_mem_float[blockDim.x + tid] += layernorm_shared_mem_float[blockDim.x + tid + s];
        }
        __syncthreads();
    }
    float mean = layernorm_shared_mem_float[0] / norm_size;
    float var = layernorm_shared_mem_float[blockDim.x] / norm_size - mean * mean;
    __syncthreads();

    // 3. 并行归一化写回
    for (int i = tid; i < norm_size; i += blockDim.x) {
        float v = in_ptr[i];
        float norm = (v - mean) / sqrt(var + eps);
        out_ptr[i] = norm * gamma[i] + beta[i];
    }
}

// 为double类型创建独立的kernel函数
__global__ void layernorm_forward_parallel_kernel_double(const double* input, double* output,
                                                         const double* gamma, const double* beta,
                                                         int batch, int norm_size, int norm_stride, double eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= batch) return;
    const double* in_ptr = input + row * norm_stride;
    double* out_ptr = output + row * norm_stride;
    extern __shared__ double layernorm_shared_mem_double[]; // layernorm_shared_mem_double[0:blockDim.x] for sum, [blockDim.x:2*blockDim.x] for sqsum

    // 1. 每个线程处理部分元素
    double sum = 0, sqsum = 0;
    for (int i = tid; i < norm_size; i += blockDim.x) {
        double v = in_ptr[i];
        sum += v;
        sqsum += v * v;
    }
    layernorm_shared_mem_double[tid] = sum;
    layernorm_shared_mem_double[blockDim.x + tid] = sqsum;
    __syncthreads();

    // 2. block内归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            layernorm_shared_mem_double[tid] += layernorm_shared_mem_double[tid + s];
            layernorm_shared_mem_double[blockDim.x + tid] += layernorm_shared_mem_double[blockDim.x + tid + s];
        }
        __syncthreads();
    }
    double mean = layernorm_shared_mem_double[0] / norm_size;
    double var = layernorm_shared_mem_double[blockDim.x] / norm_size - mean * mean;
    __syncthreads();

    // 3. 并行归一化写回
    for (int i = tid; i < norm_size; i += blockDim.x) {
        double v = in_ptr[i];
        double norm = (v - mean) / sqrt(var + eps);
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
    size_t shared_mem = threads * 2 * sizeof(T);
    
    // 根据类型调用相应的kernel函数
    if constexpr (std::is_same_v<T, float>) {
        layernorm_forward_parallel_kernel_float<<<blocks, threads, shared_mem>>>(
            input.data(), output.data(),
            gamma.data(), beta.data(), batch, norm_size, norm_stride, eps);
    } else if constexpr (std::is_same_v<T, double>) {
        layernorm_forward_parallel_kernel_double<<<blocks, threads, shared_mem>>>(
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
