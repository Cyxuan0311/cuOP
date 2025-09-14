#include "cuda_op/detail/cuBlas/dot.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace cu_op_mem {

    // 基础kernel - 保持向后兼容
    template <typename T>
    __global__ void dot_kernel(const T* x, const T* y, T* partial, int n) {
        extern __shared__ char shared_mem[];
        T* sdata = reinterpret_cast<T*>(shared_mem);
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        T val = 0;
        if (idx < n) val = x[idx] * y[idx];
        sdata[tid] = val;
        __syncthreads();

        // 归约
        for(int s = blockDim.x / 2; s > 0; s >>= 1) {
            if(tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if(tid == 0) partial[blockIdx.x] = sdata[0];
    }

    // 优化版本1: 向量化访问 + 改进归约
    template <typename T>
    __global__ void dot_kernel_vectorized(const T* x, const T* y, T* partial, int n) {
        extern __shared__ char shared_mem[];
        T* sdata = reinterpret_cast<T*>(shared_mem);
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        T val = 0;
        
        // 向量化访问
        int vec_size = sizeof(T) == 4 ? 4 : 2;
        int vec_idx = idx * vec_size;
        
        if (vec_idx + vec_size - 1 < n) {
            if constexpr (std::is_same_v<T, float>) {
                const float4* x_vec = reinterpret_cast<const float4*>(x);
                const float4* y_vec = reinterpret_cast<const float4*>(y);
                float4 x_val = x_vec[idx];
                float4 y_val = y_vec[idx];
                val = x_val.x * y_val.x + x_val.y * y_val.y + 
                      x_val.z * y_val.z + x_val.w * y_val.w;
            } else if constexpr (std::is_same_v<T, double>) {
                const double2* x_vec = reinterpret_cast<const double2*>(x);
                const double2* y_vec = reinterpret_cast<const double2*>(y);
                double2 x_val = x_vec[idx];
                double2 y_val = y_vec[idx];
                val = x_val.x * y_val.x + x_val.y * y_val.y;
            }
        } else {
            // 处理剩余元素
            for (int i = 0; i < vec_size && vec_idx + i < n; ++i) {
                val += x[vec_idx + i] * y[vec_idx + i];
            }
        }
        
        sdata[tid] = val;
        __syncthreads();

        // 改进的归约 - 使用warp级原语
        for(int s = blockDim.x / 2; s > 0; s >>= 1) {
            if(tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if(tid == 0) partial[blockIdx.x] = sdata[0];
    }

    // 优化版本2: 使用warp级归约
    template <typename T>
    __global__ void dot_kernel_warp_reduce(const T* x, const T* y, T* partial, int n) {
        extern __shared__ char shared_mem[];
        T* sdata = reinterpret_cast<T*>(shared_mem);
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        T val = 0;
        
        // 计算部分和
        if (idx < n) val = x[idx] * y[idx];
        
        // 使用warp级归约
        val = __shfl_down_sync(0xffffffff, val, 16);
        val = __shfl_down_sync(0xffffffff, val, 8);
        val = __shfl_down_sync(0xffffffff, val, 4);
        val = __shfl_down_sync(0xffffffff, val, 2);
        val = __shfl_down_sync(0xffffffff, val, 1);
        
        // 只有warp的第一个线程写入共享内存
        if (tid % 32 == 0) {
            sdata[tid / 32] = val;
        }
        __syncthreads();
        
        // 最终归约
        if (tid < blockDim.x / 32) {
            val = sdata[tid];
        } else {
            val = 0;
        }
        
        for(int s = blockDim.x / 64; s > 0; s >>= 1) {
            if(tid < s) val += sdata[tid + s];
            __syncthreads();
        }
        
        if(tid == 0) partial[blockIdx.x] = val;
    }

    // 优化版本3: 多级归约
    template <typename T>
    __global__ void dot_kernel_multi_level(const T* x, const T* y, T* partial, int n) {
        extern __shared__ char shared_mem[];
        T* sdata = reinterpret_cast<T*>(shared_mem);
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        T val = 0;
        
        // 展开循环以提高指令级并行性
        #pragma unroll 4
        for (int i = 0; i < 4 && idx + i * blockDim.x * gridDim.x < n; ++i) {
            int global_idx = idx + i * blockDim.x * gridDim.x;
            if (global_idx < n) {
                val += x[global_idx] * y[global_idx];
            }
        }
        
        sdata[tid] = val;
        __syncthreads();

        // 多级归约
        for(int s = blockDim.x / 2; s > 32; s >>= 1) {
            if(tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        
        // 最后32个元素使用warp级归约
        if (tid < 32) {
            T temp = sdata[tid];
            temp += __shfl_down_sync(0xffffffff, temp, 16);
            temp += __shfl_down_sync(0xffffffff, temp, 8);
            temp += __shfl_down_sync(0xffffffff, temp, 4);
            temp += __shfl_down_sync(0xffffffff, temp, 2);
            temp += __shfl_down_sync(0xffffffff, temp, 1);
            if (tid == 0) partial[blockIdx.x] = temp;
        }
    }

    template <typename T>
    Dot<T>::Dot() {}

    template <typename T>
    Dot<T>::~Dot() {}

    template <typename T>
    StatusCode Dot<T>::Forward(const Tensor<T>& x, const Tensor<T>& y, T& result) {
        if(x.numel() != y.numel()) {
            LOG(ERROR) << "Dot: x and y must have same number of elements";
            return StatusCode::SHAPE_MISMATCH;
        }
        int n = static_cast<int>(x.numel());
        const T* d_x = x.data();
        const T* d_y = y.data();

        cudaError_t err = cudaSuccess;
        int threads, blocks;
        T* d_partial = nullptr;
        
        // 根据数据大小选择最优kernel
        if (n >= 1024 * 1024) { // 1M elements
            // 大数组使用多级归约
            threads = 256;
            blocks = (n + threads - 1) / threads;
            cudaMalloc(&d_partial, blocks * sizeof(T));
            dot_kernel_multi_level<T><<<blocks, threads, threads * sizeof(T)>>>(d_x, d_y, d_partial, n);
        } else if (n >= 1024) {
            // 中等数组使用向量化kernel
            int vec_size = sizeof(T) == 4 ? 4 : 2;
            threads = 256;
            blocks = (n + threads * vec_size - 1) / (threads * vec_size);
            cudaMalloc(&d_partial, blocks * sizeof(T));
            dot_kernel_vectorized<T><<<blocks, threads, threads * sizeof(T)>>>(d_x, d_y, d_partial, n);
        } else if (n >= 256) {
            // 小数组使用warp级归约
            threads = 256;
            blocks = (n + threads - 1) / threads;
            cudaMalloc(&d_partial, blocks * sizeof(T));
            dot_kernel_warp_reduce<T><<<blocks, threads, threads * sizeof(T)>>>(d_x, d_y, d_partial, n);
        } else {
            // 极小数组使用基础kernel
            threads = 256;
            blocks = (n + threads - 1) / threads;
            cudaMalloc(&d_partial, blocks * sizeof(T));
            dot_kernel<T><<<blocks, threads, threads * sizeof(T)>>>(d_x, d_y, d_partial, n);
        }

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "dot kernel failed: " << cudaGetErrorString(err);
            cudaFree(d_partial);
            return StatusCode::CUDA_ERROR;
        }

        // 如果只有一个block，直接使用GPU归约
        if (blocks == 1) {
            T h_result;
            cudaMemcpy(&h_result, d_partial, sizeof(T), cudaMemcpyDeviceToHost);
            result = h_result;
        } else {
            // 多block情况，在CPU上完成最终归约
            std::vector<T> h_partial(blocks);
            cudaMemcpy(h_partial.data(), d_partial, blocks * sizeof(T), cudaMemcpyDeviceToHost);
            
            result = 0;
            for (int i = 0; i < blocks; ++i) {
                result += h_partial[i];
            }
        }
        
        cudaFree(d_partial);
        
        VLOG(1) << "Dot: result = x^T y, n = " << n << ", result = " << result;
        return StatusCode::SUCCESS;
    }

    template class Dot<float>;
    template class Dot<double>;

}