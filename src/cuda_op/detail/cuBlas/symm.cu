#include "cuda_op/detail/cuBlas/symm.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace cu_op_mem {

// 基础kernel - 保持向后兼容
template <typename T>
__global__ void symm_kernel(int m, int n, T alpha, const T* A, const T* B, T beta, T* C, bool left, bool upper) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        T sum = 0;
        if (left) {
            for (int k = 0; k < m; ++k) {
                T a = upper ? (row <= k ? A[row * m + k] : A[k * m + row]) : (row >= k ? A[row * m + k] : A[k * m + row]);
                sum += a * B[k * n + col];
            }
        } else {
            for (int k = 0; k < n; ++k) {
                T a = upper ? (col <= k ? A[col * n + k] : A[k * n + col]) : (col >= k ? A[col * n + k] : A[k * n + col]);
                sum += B[row * n + k] * a;
            }
        }
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}

// 优化版本1: 共享内存优化
template <typename T, int TILE_SIZE>
__global__ void symm_kernel_shared(int m, int n, T alpha, const T* A, const T* B, T beta, T* C, bool left, bool upper) {
    __shared__ T shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ T shared_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    
    T sum = 0;
    
    if (left) {
        // C = A * B, A是m×m对称矩阵
        for (int tile = 0; tile < (m + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
            int k = tile * TILE_SIZE + tid_x;
            int l = tile * TILE_SIZE + tid_y;
            
            // 协作加载A到共享内存
            if (row < m && k < m) {
                T a_val = upper ? (row <= k ? A[row * m + k] : A[k * m + row]) : 
                                 (row >= k ? A[row * m + k] : A[k * m + row]);
                shared_A[tid_y][tid_x] = a_val;
            } else {
                shared_A[tid_y][tid_x] = 0;
            }
            
            // 协作加载B到共享内存
            if (l < m && col < n) {
                shared_B[tid_y][tid_x] = B[l * n + col];
            } else {
                shared_B[tid_y][tid_x] = 0;
            }
            
            __syncthreads();
            
            // 计算部分和
            if (row < m && col < n) {
                for (int i = 0; i < TILE_SIZE && tile * TILE_SIZE + i < m; ++i) {
                    sum += shared_A[tid_y][i] * shared_B[i][tid_x];
                }
            }
            __syncthreads();
        }
    } else {
        // C = B * A, A是n×n对称矩阵
        for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
            int k = tile * TILE_SIZE + tid_x;
            int l = tile * TILE_SIZE + tid_y;
            
            // 协作加载A到共享内存
            if (col < n && k < n) {
                T a_val = upper ? (col <= k ? A[col * n + k] : A[k * n + col]) : 
                                 (col >= k ? A[col * n + k] : A[k * n + col]);
                shared_A[tid_y][tid_x] = a_val;
            } else {
                shared_A[tid_y][tid_x] = 0;
            }
            
            // 协作加载B到共享内存
            if (row < m && l < n) {
                shared_B[tid_y][tid_x] = B[row * n + l];
            } else {
                shared_B[tid_y][tid_x] = 0;
            }
            
            __syncthreads();
            
            // 计算部分和
            if (row < m && col < n) {
                for (int i = 0; i < TILE_SIZE && tile * TILE_SIZE + i < n; ++i) {
                    sum += shared_B[tid_y][i] * shared_A[i][tid_x];
                }
            }
            __syncthreads();
        }
    }
    
    if (row < m && col < n) {
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}

// 优化版本2: 向量化访问
template <typename T>
__global__ void symm_kernel_vectorized(int m, int n, T alpha, const T* A, const T* B, T beta, T* C, bool left, bool upper) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        T sum = 0;
        
        if (left) {
            // 向量化访问B
            int vec_size = sizeof(T) == 4 ? 4 : 2;
            int vec_m = (m / vec_size) * vec_size;
            
            // 向量化部分
            for (int k = 0; k < vec_m; k += vec_size) {
                if constexpr (std::is_same_v<T, float>) {
                    const float4* B_vec = reinterpret_cast<const float4*>(B);
                    float4 b_val = B_vec[k * n / 4 + col];
                    
                    for (int i = 0; i < 4; ++i) {
                        T a = upper ? (row <= k + i ? A[row * m + k + i] : A[(k + i) * m + row]) : 
                                     (row >= k + i ? A[row * m + k + i] : A[(k + i) * m + row]);
                        sum += a * (&b_val.x)[i];
                    }
                } else if constexpr (std::is_same_v<T, double>) {
                    const double2* B_vec = reinterpret_cast<const double2*>(B);
                    double2 b_val = B_vec[k * n / 2 + col];
                    
                    for (int i = 0; i < 2; ++i) {
                        T a = upper ? (row <= k + i ? A[row * m + k + i] : A[(k + i) * m + row]) : 
                                     (row >= k + i ? A[row * m + k + i] : A[(k + i) * m + row]);
                        sum += a * (&b_val.x)[i];
                    }
                }
            }
            
            // 处理剩余元素
            for (int k = vec_m; k < m; ++k) {
                T a = upper ? (row <= k ? A[row * m + k] : A[k * m + row]) : 
                             (row >= k ? A[row * m + k] : A[k * m + row]);
                sum += a * B[k * n + col];
            }
        } else {
            // 类似地处理C = B * A的情况
            int vec_size = sizeof(T) == 4 ? 4 : 2;
            int vec_n = (n / vec_size) * vec_size;
            
            // 向量化部分
            for (int k = 0; k < vec_n; k += vec_size) {
                if constexpr (std::is_same_v<T, float>) {
                    const float4* B_vec = reinterpret_cast<const float4*>(B);
                    float4 b_val = B_vec[row * n / 4 + k / 4];
                    
                    for (int i = 0; i < 4; ++i) {
                        T a = upper ? (col <= k + i ? A[col * n + k + i] : A[(k + i) * n + col]) : 
                                     (col >= k + i ? A[col * n + k + i] : A[(k + i) * n + col]);
                        sum += (&b_val.x)[i] * a;
                    }
                } else if constexpr (std::is_same_v<T, double>) {
                    const double2* B_vec = reinterpret_cast<const double2*>(B);
                    double2 b_val = B_vec[row * n / 2 + k / 2];
                    
                    for (int i = 0; i < 2; ++i) {
                        T a = upper ? (col <= k + i ? A[col * n + k + i] : A[(k + i) * n + col]) : 
                                     (col >= k + i ? A[col * n + k + i] : A[(k + i) * n + col]);
                        sum += (&b_val.x)[i] * a;
                    }
                }
            }
            
            // 处理剩余元素
            for (int k = vec_n; k < n; ++k) {
                T a = upper ? (col <= k ? A[col * n + k] : A[k * n + col]) : 
                             (col >= k ? A[col * n + k] : A[k * n + col]);
                sum += B[row * n + k] * a;
            }
        }
        
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}

template <typename T>
Symm<T>::Symm(cublasSideMode_t side, cublasFillMode_t uplo, T alpha, T beta)
    : side_(side), uplo_(uplo), alpha_(alpha), beta_(beta) {}

template <typename T>
Symm<T>::~Symm() {}

template <typename T>
void Symm<T>::SetWeight(const Tensor<T>& weight) {
    weight_ = Tensor<T>(weight.shape());
    cudaMemcpy(weight_.data(), weight.data(), weight.bytes(), cudaMemcpyDeviceToDevice);
    VLOG(1) << "Symm weight set, shape=[" << weight.shape()[0] << ", " << weight.shape()[1] << "]";
}

template <typename T>
StatusCode Symm<T>::Forward(const Tensor<T>& input, Tensor<T>& output) {
    int m = static_cast<int>(input.shape()[0]);
    int n = static_cast<int>(input.shape()[1]);
    if (weight_.shape()[0] != m || weight_.shape()[1] != m || output.shape()[0] != m || output.shape()[1] != n) {
        LOG(ERROR) << "Symm: shape mismatch";
        return StatusCode::SHAPE_MISMATCH;
    }
    const T* d_A = weight_.data();
    const T* d_B = input.data();
    T* d_C = output.data();

    cudaError_t err = cudaSuccess;
    bool left = (side_ == 0); // CUBLAS_SIDE_LEFT
    bool upper = (uplo_ == 0); // CUBLAS_FILL_MODE_UPPER
    
    // 根据矩阵大小选择最优kernel
    if (m >= 512 && n >= 512) {
        // 大矩阵使用共享内存优化
        constexpr int TILE_SIZE = 32;
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
        symm_kernel_shared<T, TILE_SIZE><<<blocks, threads>>>(m, n, alpha_, d_A, d_B, beta_, d_C, left, upper);
    } else if (m >= 128 && n >= 128) {
        // 中等矩阵使用向量化kernel
        dim3 threads(16, 16);
        dim3 blocks((n + 15) / 16, (m + 15) / 16);
        symm_kernel_vectorized<T><<<blocks, threads>>>(m, n, alpha_, d_A, d_B, beta_, d_C, left, upper);
    } else {
        // 小矩阵使用基础kernel
        dim3 threads(16, 16);
        dim3 blocks((n + 15) / 16, (m + 15) / 16);
        symm_kernel<T><<<blocks, threads>>>(m, n, alpha_, d_A, d_B, beta_, d_C, left, upper);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "symm kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    
    VLOG(1) << "Symm: C = " << alpha_ << " * A * B + " << beta_ << " * C, m = " << m << ", n = " << n;
    return StatusCode::SUCCESS;
}

// 显式实例化
template class Symm<float>;
template class Symm<double>;

} // namespace cu_op_mem