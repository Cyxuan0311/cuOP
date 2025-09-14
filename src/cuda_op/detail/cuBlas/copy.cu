#include "cuda_op/detail/cuBlas/copy.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace cu_op_mem {
    // 基础kernel - 保持向后兼容
    template <typename T>
    __global__ void copy_kernel(int n, const T* x, T* y) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < n) {
            y[idx] = x[idx];
        }
    }

    // 优化版本1: 向量化访问 (float4/double2)
    template <typename T>
    __global__ void copy_kernel_vectorized(int n, const T* x, T* y) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int vec_size = sizeof(T) == 4 ? 4 : 2; // float4 or double2
        int vec_idx = idx * vec_size;
        
        if (vec_idx + vec_size - 1 < n) {
            if constexpr (std::is_same_v<T, float>) {
                const float4* x_vec = reinterpret_cast<const float4*>(x);
                float4* y_vec = reinterpret_cast<float4*>(y);
                y_vec[idx] = x_vec[idx];
            } else if constexpr (std::is_same_v<T, double>) {
                const double2* x_vec = reinterpret_cast<const double2*>(x);
                double2* y_vec = reinterpret_cast<double2*>(y);
                y_vec[idx] = x_vec[idx];
            }
        } else {
            // 处理剩余元素
            for (int i = 0; i < vec_size && vec_idx + i < n; ++i) {
                y[vec_idx + i] = x[vec_idx + i];
            }
        }
    }

    // 优化版本2: 共享内存 + 向量化
    template <typename T, int BLOCK_SIZE>
    __global__ void copy_kernel_shared(int n, const T* x, T* y) {
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
                const float4* shared_x_vec = reinterpret_cast<const float4*>(shared_data);
                float4* shared_y_vec = reinterpret_cast<float4*>(shared_data);
                shared_y_vec[tid] = shared_x_vec[tid];
            } else if constexpr (std::is_same_v<T, double>) {
                const double2* shared_x_vec = reinterpret_cast<const double2*>(shared_data);
                double2* shared_y_vec = reinterpret_cast<double2*>(shared_data);
                shared_y_vec[tid] = shared_x_vec[tid];
            }
        } else {
            // 标量处理剩余元素
            if (tid < BLOCK_SIZE && idx < n) {
                shared_data[tid] = shared_data[tid]; // 已经在共享内存中
            }
        }
        __syncthreads();
        
        // 写回全局内存
        if (idx < n) {
            y[idx] = shared_data[tid];
        }
    }

    // 优化版本3: 融合内存访问
    template <typename T>
    __global__ void copy_kernel_fused(int n, const T* x, T* y) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        // 展开循环以提高指令级并行性
        #pragma unroll 4
        for (int i = 0; i < 4 && idx + i * blockDim.x * gridDim.x < n; ++i) {
            int global_idx = idx + i * blockDim.x * gridDim.x;
            if (global_idx < n) {
                y[global_idx] = x[global_idx];
            }
        }
    }

    template <typename T>
    Copy<T>::Copy() {}

    template <typename T>
    Copy<T>::~Copy() {}

    template <typename T>
    StatusCode Copy<T>::Forward(const Tensor<T>& x, Tensor<T>& y) {
        if(x.numel() != y.numel()) {
            LOG(ERROR) << "Copy: x and y must have the same number of elements";
            return StatusCode::SHAPE_MISMATCH;
        }
        int n = static_cast<int>(x.numel());
        const T* d_x = x.data();
        T* d_y = y.data();

        cudaError_t err = cudaSuccess;
        size_t bytes = n * sizeof(T);
        
        // 对于大数组，优先使用cudaMemcpyAsync
        if (n >= 1024 * 1024) { // 1M elements
            err = cudaMemcpyAsync(d_y, d_x, bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                LOG(ERROR) << "cudaMemcpyAsync failed: " << cudaGetErrorString(err);
                return StatusCode::CUDA_ERROR;
            }
        } else if (n >= 1024) {
            // 大数组使用向量化kernel
            int vec_size = sizeof(T) == 4 ? 4 : 2;
            int threads = 256;
            int blocks = (n + threads * vec_size - 1) / (threads * vec_size);
            copy_kernel_vectorized<T><<<blocks, threads>>>(n, d_x, d_y);
        } else if (n >= 256) {
            // 中等数组使用共享内存优化
            constexpr int BLOCK_SIZE = 256;
            int threads = BLOCK_SIZE;
            int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            copy_kernel_shared<T, BLOCK_SIZE><<<blocks, threads>>>(n, d_x, d_y);
        } else if (n >= 64) {
            // 小数组使用融合kernel
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            copy_kernel_fused<T><<<blocks, threads>>>(n, d_x, d_y);
        } else {
            // 极小数组使用基础kernel
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            copy_kernel<T><<<blocks, threads>>>(n, d_x, d_y);
        }

        if (n < 1024 * 1024) { // 只有使用kernel时才检查错误
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                LOG(ERROR) << "copy kernel failed: " << cudaGetErrorString(err);
                return StatusCode::CUDA_ERROR;
            }
        }
        
        VLOG(1) << "Copy: y = x, n = " << n;
        return StatusCode::SUCCESS;
    }

    template class Copy<float>;
    template class Copy<double>;
    
}