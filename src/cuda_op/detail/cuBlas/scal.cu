#include "cuda_op/detail/cuBlas/scal.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace cu_op_mem {

// 基础kernel - 保持向后兼容
template <typename T>
__global__ void scal_kernel(int n, T alpha, T* x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        x[idx] = alpha * x[idx];
    }
}

// 优化版本1: 向量化访问 (float4/double2)
template <typename T>
__global__ void scal_kernel_vectorized(int n, T alpha, T* x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_size = sizeof(T) == 4 ? 4 : 2; // float4 or double2
    int vec_idx = idx * vec_size;
    
    if (vec_idx + vec_size - 1 < n) {
        if constexpr (std::is_same_v<T, float>) {
            float4* x_vec = reinterpret_cast<float4*>(x);
            float4 val = x_vec[idx];
            val.x *= alpha;
            val.y *= alpha;
            val.z *= alpha;
            val.w *= alpha;
            x_vec[idx] = val;
        } else if constexpr (std::is_same_v<T, double>) {
            double2* x_vec = reinterpret_cast<double2*>(x);
            double2 val = x_vec[idx];
            val.x *= alpha;
            val.y *= alpha;
            x_vec[idx] = val;
        }
    } else {
        // 处理剩余元素
        for (int i = 0; i < vec_size && vec_idx + i < n; ++i) {
            x[vec_idx + i] *= alpha;
        }
    }
}

// 优化版本2: 共享内存 + 向量化
template <typename T, int BLOCK_SIZE>
__global__ void scal_kernel_shared(int n, T alpha, T* x) {
    __shared__ T shared_data[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;
    
    // 协作加载到共享内存
    if (idx < n) {
        shared_data[tid] = x[idx];
    }
    __syncthreads();
    
    // 向量化处理
    int vec_size = sizeof(T) == 4 ? 4 : 2;
    int vec_tid = tid * vec_size;
    
    if (vec_tid + vec_size - 1 < BLOCK_SIZE && idx + vec_size - 1 < n) {
        if constexpr (std::is_same_v<T, float>) {
            float4* shared_vec = reinterpret_cast<float4*>(shared_data);
            float4 val = shared_vec[tid];
            val.x *= alpha;
            val.y *= alpha;
            val.z *= alpha;
            val.w *= alpha;
            shared_vec[tid] = val;
        } else if constexpr (std::is_same_v<T, double>) {
            double2* shared_vec = reinterpret_cast<double2*>(shared_data);
            double2 val = shared_vec[tid];
            val.x *= alpha;
            val.y *= alpha;
            shared_vec[tid] = val;
        }
    } else {
        // 标量处理剩余元素
        if (tid < BLOCK_SIZE && idx < n) {
            shared_data[tid] *= alpha;
        }
    }
    __syncthreads();
    
    // 写回全局内存
    if (idx < n) {
        x[idx] = shared_data[tid];
    }
}

// 优化版本3: 多流并行版本
template <typename T>
__global__ void scal_kernel_multi_stream(int n, T alpha, T* x, int stream_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + stream_offset;
    if(idx < n) {
        x[idx] = alpha * x[idx];
    }
}

template <typename T>
Scal<T>::Scal(T alpha) : alpha_(alpha) {}

template <typename T>
Scal<T>::~Scal() = default;

template <typename T>
void Scal<T>::SetAlpha(T alpha) {
    alpha_ = alpha;
}

template <typename T>
StatusCode Scal<T>::Forward(Tensor<T>& x) {
    int n = static_cast<int>(x.numel());
    T* d_x = x.data();

    cudaError_t err = cudaSuccess;
    
    // 根据数据大小选择最优kernel
    if (n >= 1024) {
        // 大数组使用向量化kernel
        int vec_size = sizeof(T) == 4 ? 4 : 2;
        int threads = 256;
        int blocks = (n + threads * vec_size - 1) / (threads * vec_size);
        scal_kernel_vectorized<T><<<blocks, threads>>>(n, alpha_, d_x);
    } else if (n >= 256) {
        // 中等数组使用共享内存优化
        constexpr int BLOCK_SIZE = 256;
        int threads = BLOCK_SIZE;
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scal_kernel_shared<T, BLOCK_SIZE><<<blocks, threads>>>(n, alpha_, d_x);
    } else {
        // 小数组使用基础kernel
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        scal_kernel<T><<<blocks, threads>>>(n, alpha_, d_x);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "scal kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    
    VLOG(1) << "Scal: x = " << alpha_ << " * x, n = " << n;
    return StatusCode::SUCCESS;
}

template class Scal<float>;
template class Scal<double>;

}