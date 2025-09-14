#include "cuda_op/detail/cuBlas/gemv.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace cu_op_mem {

// 基础kernel - 保持向后兼容
template <typename T>
__global__ void gemv_kernel(int m, int n, T alpha, const T* A, const T* x, T beta, T* y, bool transA) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        T sum = 0;
        for (int col = 0; col < n; ++col) {
            if (transA)
                sum += A[col * m + row] * x[col];
            else
                sum += A[row * n + col] * x[col];
        }
        y[row] = alpha * sum + beta * y[row];
    }
}

// 优化版本1: 共享内存优化
template <typename T, int TILE_SIZE>
__global__ void gemv_kernel_shared(int m, int n, T alpha, const T* A, const T* x, T beta, T* y, bool transA) {
    __shared__ T shared_x[TILE_SIZE];
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    T sum = 0;
    
    // 分块处理
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int col = tile * TILE_SIZE + tid;
        
        // 协作加载x到共享内存
        if (col < n) {
            shared_x[tid] = x[col];
        } else {
            shared_x[tid] = 0;
        }
        __syncthreads();
        
        // 计算部分和
        if (row < m) {
            for (int i = 0; i < TILE_SIZE && tile * TILE_SIZE + i < n; ++i) {
                if (transA) {
                    sum += A[(tile * TILE_SIZE + i) * m + row] * shared_x[i];
                } else {
                    sum += A[row * n + tile * TILE_SIZE + i] * shared_x[i];
                }
            }
        }
        __syncthreads();
    }
    
    if (row < m) {
        y[row] = alpha * sum + beta * y[row];
    }
}

// 优化版本2: 向量化访问
template <typename T>
__global__ void gemv_kernel_vectorized(int m, int n, T alpha, const T* A, const T* x, T beta, T* y, bool transA) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        T sum = 0;
        
        // 向量化访问
        int vec_size = sizeof(T) == 4 ? 4 : 2;
        int vec_n = (n / vec_size) * vec_size;
        
        // 向量化部分
        for (int col = 0; col < vec_n; col += vec_size) {
            if constexpr (std::is_same_v<T, float>) {
                const float4* x_vec = reinterpret_cast<const float4*>(x);
                const float4* A_vec = reinterpret_cast<const float4*>(A);
                float4 x_val = x_vec[col / 4];
                
                if (transA) {
                    for (int i = 0; i < 4; ++i) {
                        sum += A[(col + i) * m + row] * (&x_val.x)[i];
                    }
                } else {
                    float4 A_val = A_vec[row * n / 4 + col / 4];
                    sum += A_val.x * x_val.x + A_val.y * x_val.y + 
                           A_val.z * x_val.z + A_val.w * x_val.w;
                }
            } else if constexpr (std::is_same_v<T, double>) {
                const double2* x_vec = reinterpret_cast<const double2*>(x);
                const double2* A_vec = reinterpret_cast<const double2*>(A);
                double2 x_val = x_vec[col / 2];
                
                if (transA) {
                    for (int i = 0; i < 2; ++i) {
                        sum += A[(col + i) * m + row] * (&x_val.x)[i];
                    }
                } else {
                    double2 A_val = A_vec[row * n / 2 + col / 2];
                    sum += A_val.x * x_val.x + A_val.y * x_val.y;
                }
            }
        }
        
        // 处理剩余元素
        for (int col = vec_n; col < n; ++col) {
            if (transA)
                sum += A[col * m + row] * x[col];
            else
                sum += A[row * n + col] * x[col];
        }
        
        y[row] = alpha * sum + beta * y[row];
    }
}

// 优化版本3: 分块 + 向量化
template <typename T, int TILE_SIZE>
__global__ void gemv_kernel_tiled_vectorized(int m, int n, T alpha, const T* A, const T* x, T beta, T* y, bool transA) {
    __shared__ T shared_x[TILE_SIZE];
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    T sum = 0;
    
    // 分块处理
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int col = tile * TILE_SIZE + tid;
        
        // 协作加载x到共享内存
        if (col < n) {
            shared_x[tid] = x[col];
        } else {
            shared_x[tid] = 0;
        }
        __syncthreads();
        
        // 向量化计算
        if (row < m) {
            int vec_size = sizeof(T) == 4 ? 4 : 2;
            int vec_tile_size = (TILE_SIZE / vec_size) * vec_size;
            
            // 向量化部分
            for (int i = 0; i < vec_tile_size; i += vec_size) {
                if (tile * TILE_SIZE + i + vec_size - 1 < n) {
                    if constexpr (std::is_same_v<T, float>) {
                        const float4* shared_x_vec = reinterpret_cast<const float4*>(shared_x);
                        float4 x_val = shared_x_vec[i / 4];
                        
                        if (transA) {
                            for (int j = 0; j < 4; ++j) {
                                sum += A[(tile * TILE_SIZE + i + j) * m + row] * (&x_val.x)[j];
                            }
                        } else {
                            const float4* A_vec = reinterpret_cast<const float4*>(A);
                            float4 A_val = A_vec[row * n / 4 + (tile * TILE_SIZE + i) / 4];
                            sum += A_val.x * x_val.x + A_val.y * x_val.y + 
                                   A_val.z * x_val.z + A_val.w * x_val.w;
                        }
                    } else if constexpr (std::is_same_v<T, double>) {
                        const double2* shared_x_vec = reinterpret_cast<const double2*>(shared_x);
                        double2 x_val = shared_x_vec[i / 2];
                        
                        if (transA) {
                            for (int j = 0; j < 2; ++j) {
                                sum += A[(tile * TILE_SIZE + i + j) * m + row] * (&x_val.x)[j];
                            }
                        } else {
                            const double2* A_vec = reinterpret_cast<const double2*>(A);
                            double2 A_val = A_vec[row * n / 2 + (tile * TILE_SIZE + i) / 2];
                            sum += A_val.x * x_val.x + A_val.y * x_val.y;
                        }
                    }
                }
            }
            
            // 处理剩余元素
            for (int i = vec_tile_size; i < TILE_SIZE && tile * TILE_SIZE + i < n; ++i) {
                if (transA) {
                    sum += A[(tile * TILE_SIZE + i) * m + row] * shared_x[i];
                } else {
                    sum += A[row * n + tile * TILE_SIZE + i] * shared_x[i];
                }
            }
        }
        __syncthreads();
    }
    
    if (row < m) {
        y[row] = alpha * sum + beta * y[row];
    }
}

template <typename T>
Gemv<T>::Gemv(bool transA, T alpha, T beta)
    : transA_(transA), alpha_(alpha), beta_(beta) {}

template <typename T>
Gemv<T>::~Gemv() {}

template <typename T>
void Gemv<T>::SetWeight(const Tensor<T>& weight) {
    weight_ = Tensor<T>(weight.shape());
    cudaMemcpy(weight_.data(), weight.data(), weight.bytes(), cudaMemcpyDeviceToDevice);
}

template <typename T>
StatusCode Gemv<T>::Forward(const Tensor<T>& input, Tensor<T>& output) {
    int m = static_cast<int>(input.shape()[0]);
    int n = static_cast<int>(input.shape()[1]);
    if (weight_.numel() != n || output.numel() != m) {
        LOG(ERROR) << "Gemv: shape mismatch";
        return StatusCode::SHAPE_MISMATCH;
    }
    const T* d_A = input.data();
    const T* d_x = weight_.data();
    T* d_y = output.data();

    cudaError_t err = cudaSuccess;
    
    // 根据矩阵大小选择最优kernel
    if (m >= 1024 && n >= 1024) {
        // 大矩阵使用分块+向量化kernel
        constexpr int TILE_SIZE = 256;
        int threads = 256;
        int blocks = (m + threads - 1) / threads;
        gemv_kernel_tiled_vectorized<T, TILE_SIZE><<<blocks, threads>>>(m, n, alpha_, d_A, d_x, beta_, d_y, transA_);
    } else if (m >= 256 && n >= 256) {
        // 中等矩阵使用共享内存优化
        constexpr int TILE_SIZE = 128;
        int threads = 256;
        int blocks = (m + threads - 1) / threads;
        gemv_kernel_shared<T, TILE_SIZE><<<blocks, threads>>>(m, n, alpha_, d_A, d_x, beta_, d_y, transA_);
    } else if (m >= 64 && n >= 64) {
        // 小矩阵使用向量化kernel
        int threads = 256;
        int blocks = (m + threads - 1) / threads;
        gemv_kernel_vectorized<T><<<blocks, threads>>>(m, n, alpha_, d_A, d_x, beta_, d_y, transA_);
    } else {
        // 极小矩阵使用基础kernel
        int threads = 256;
        int blocks = (m + threads - 1) / threads;
        gemv_kernel<T><<<blocks, threads>>>(m, n, alpha_, d_A, d_x, beta_, d_y, transA_);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "gemv kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    
    VLOG(1) << "Gemv: y = " << alpha_ << " * A * x + " << beta_ << " * y, m = " << m << ", n = " << n;
    return StatusCode::SUCCESS;
}

// 显式实例化
template class Gemv<float>;
template class Gemv<double>;

} // namespace cu_op_mem