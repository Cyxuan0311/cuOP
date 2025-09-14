#include "cuda_op/detail/cuBlas/axpy.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace cu_op_mem {

// 基础kernel - 保持向后兼容
template <typename T>
__global__ void axpy_kernel(int n, T alpha, const T* x, T* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

// 优化版本1: 向量化访问 (float4/double2)
template <typename T>
__global__ void axpy_kernel_vectorized(int n, T alpha, const T* x, T* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_size = sizeof(T) == 4 ? 4 : 2; // float4 or double2
    int vec_idx = idx * vec_size;
    
    if (vec_idx + vec_size - 1 < n) {
        if constexpr (std::is_same_v<T, float>) {
            const float4* x_vec = reinterpret_cast<const float4*>(x);
            float4* y_vec = reinterpret_cast<float4*>(y);
            float4 x_val = x_vec[idx];
            float4 y_val = y_vec[idx];
            y_val.x = alpha * x_val.x + y_val.x;
            y_val.y = alpha * x_val.y + y_val.y;
            y_val.z = alpha * x_val.z + y_val.z;
            y_val.w = alpha * x_val.w + y_val.w;
            y_vec[idx] = y_val;
        } else if constexpr (std::is_same_v<T, double>) {
            const double2* x_vec = reinterpret_cast<const double2*>(x);
            double2* y_vec = reinterpret_cast<double2*>(y);
            double2 x_val = x_vec[idx];
            double2 y_val = y_vec[idx];
            y_val.x = alpha * x_val.x + y_val.x;
            y_val.y = alpha * x_val.y + y_val.y;
            y_vec[idx] = y_val;
        }
    } else {
        // 处理剩余元素
        for (int i = 0; i < vec_size && vec_idx + i < n; ++i) {
            y[vec_idx + i] = alpha * x[vec_idx + i] + y[vec_idx + i];
        }
    }
}

// 优化版本2: 共享内存 + 向量化
template <typename T, int BLOCK_SIZE>
__global__ void axpy_kernel_shared(int n, T alpha, const T* x, T* y) {
    __shared__ T shared_x[BLOCK_SIZE];
    __shared__ T shared_y[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;
    
    // 协作加载到共享内存
    if (idx < n) {
        shared_x[tid] = x[idx];
        shared_y[tid] = y[idx];
    }
    __syncthreads();
    
    // 向量化处理
    int vec_size = sizeof(T) == 4 ? 4 : 2;
    int vec_tid = tid * vec_size;
    
    if (vec_tid + vec_size - 1 < BLOCK_SIZE && idx + vec_size - 1 < n) {
        if constexpr (std::is_same_v<T, float>) {
            const float4* shared_x_vec = reinterpret_cast<const float4*>(shared_x);
            float4* shared_y_vec = reinterpret_cast<float4*>(shared_y);
            float4 x_val = shared_x_vec[tid];
            float4 y_val = shared_y_vec[tid];
            y_val.x = alpha * x_val.x + y_val.x;
            y_val.y = alpha * x_val.y + y_val.y;
            y_val.z = alpha * x_val.z + y_val.z;
            y_val.w = alpha * x_val.w + y_val.w;
            shared_y_vec[tid] = y_val;
        } else if constexpr (std::is_same_v<T, double>) {
            const double2* shared_x_vec = reinterpret_cast<const double2*>(shared_x);
            double2* shared_y_vec = reinterpret_cast<double2*>(shared_y);
            double2 x_val = shared_x_vec[tid];
            double2 y_val = shared_y_vec[tid];
            y_val.x = alpha * x_val.x + y_val.x;
            y_val.y = alpha * x_val.y + y_val.y;
            shared_y_vec[tid] = y_val;
        }
    } else {
        // 标量处理剩余元素
        if (tid < BLOCK_SIZE && idx < n) {
            shared_y[tid] = alpha * shared_x[tid] + shared_y[tid];
        }
    }
    __syncthreads();
    
    // 写回全局内存
    if (idx < n) {
        y[idx] = shared_y[tid];
    }
}

// 优化版本3: 融合内存访问
template <typename T>
__global__ void axpy_kernel_fused(int n, T alpha, const T* x, T* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 展开循环以提高指令级并行性
    #pragma unroll 4
    for (int i = 0; i < 4 && idx + i * blockDim.x * gridDim.x < n; ++i) {
        int global_idx = idx + i * blockDim.x * gridDim.x;
        if (global_idx < n) {
            y[global_idx] = alpha * x[global_idx] + y[global_idx];
        }
    }
}

template <typename T>
Axpy<T>::Axpy(T alpha) : alpha_(alpha) {}

template <typename T>
Axpy<T>::~Axpy() {}

template <typename T>
void Axpy<T>::SetAlpha(T alpha) {
    alpha_ = alpha;
}

template <typename T>
StatusCode Axpy<T>::Forward(const Tensor<T>& x, Tensor<T>& y) {
    if (x.numel() != y.numel()) {
        LOG(ERROR) << "Axpy: x and y must have the same number of elements";
        return StatusCode::SHAPE_MISMATCH;
    }
    int n = static_cast<int>(x.numel());
    const T* d_x = x.data();
    T* d_y = y.data();

    cudaError_t err = cudaSuccess;
    
    // 根据数据大小选择最优kernel
    if (n >= 2048) {
        // 大数组使用向量化kernel
        int vec_size = sizeof(T) == 4 ? 4 : 2;
        int threads = 256;
        int blocks = (n + threads * vec_size - 1) / (threads * vec_size);
        axpy_kernel_vectorized<T><<<blocks, threads>>>(n, alpha_, d_x, d_y);
    } else if (n >= 512) {
        // 中等数组使用共享内存优化
        constexpr int BLOCK_SIZE = 256;
        int threads = BLOCK_SIZE;
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        axpy_kernel_shared<T, BLOCK_SIZE><<<blocks, threads>>>(n, alpha_, d_x, d_y);
    } else if (n >= 64) {
        // 小数组使用融合kernel
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        axpy_kernel_fused<T><<<blocks, threads>>>(n, alpha_, d_x, d_y);
    } else {
        // 极小数组使用基础kernel
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        axpy_kernel<T><<<blocks, threads>>>(n, alpha_, d_x, d_y);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "axpy kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    
    VLOG(1) << "Axpy: y = " << alpha_ << " * x + y, n = " << n;
    return StatusCode::SUCCESS;
}

// 显式实例化
template class Axpy<float>;
template class Axpy<double>;

} // namespace cu_op_mem