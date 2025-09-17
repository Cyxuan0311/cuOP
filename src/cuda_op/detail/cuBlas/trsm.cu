#include "cuda_op/detail/cuBlas/trsm.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace cu_op_mem {

// 基础kernel - 保持向后兼容
template <typename T>
__global__ void trsm_left_lower_kernel(int m, int n, T alpha, const T* A, T* B) {
    // A: m x m lower-triangular, B: m x n, solve AX = alpha*B, overwrite B with X
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        for (int row = 0; row < m; ++row) {
            T sum = alpha * B[row * n + col];
            for (int k = 0; k < row; ++k) {
                sum -= A[row * m + k] * B[k * n + col];
            }
            B[row * n + col] = sum / A[row * m + row];
        }
    }
}

// 优化版本1: 并行化行处理 - float特化
__global__ void trsm_left_lower_parallel_kernel_float(int m, int n, float alpha, const float* A, float* B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < n && row < m) {
        float sum = alpha * B[row * n + col];
        
        // 使用共享内存存储B的列
        extern __shared__ float shared_B_parallel_data[];
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int shared_size = blockDim.x * blockDim.y;
        
        // 协作加载B的列到共享内存
        for (int i = 0; i < m; i += shared_size) {
            int idx = i + tid;
            if (idx < m && tid < shared_size) {
                shared_B_parallel_data[tid] = B[idx * n + col];
            }
            __syncthreads();
            
            // 计算部分和
            for (int k = 0; k < blockDim.y && i + k < row; ++k) {
                int k_idx = i + k;
                int shared_idx = k * blockDim.x + threadIdx.x;
                if (k_idx < row && shared_idx < shared_size) {
                    sum -= A[row * m + k_idx] * shared_B_parallel_data[shared_idx];
                }
            }
            __syncthreads();
        }
        
        B[row * n + col] = sum / A[row * m + row];
    }
}

// 优化版本1: 并行化行处理 - double特化
__global__ void trsm_left_lower_parallel_kernel_double(int m, int n, double alpha, const double* A, double* B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < n && row < m) {
        double sum = alpha * B[row * n + col];
        
        // 使用共享内存存储B的列
        extern __shared__ double shared_B_parallel_data_double[];
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int shared_size = blockDim.x * blockDim.y;
        
        // 协作加载B的列到共享内存
        for (int i = 0; i < m; i += shared_size) {
            int idx = i + tid;
            if (idx < m && tid < shared_size) {
                shared_B_parallel_data_double[tid] = B[idx * n + col];
            }
            __syncthreads();
            
            // 计算部分和
            for (int k = 0; k < blockDim.y && i + k < row; ++k) {
                int k_idx = i + k;
                int shared_idx = k * blockDim.x + threadIdx.x;
                if (k_idx < row && shared_idx < shared_size) {
                    sum -= A[row * m + k_idx] * shared_B_parallel_data_double[shared_idx];
                }
            }
            __syncthreads();
        }
        
        B[row * n + col] = sum / A[row * m + row];
    }
}

// 优化版本2: 分块处理
template <typename T, int BLOCK_SIZE>
__global__ void trsm_left_lower_blocked_kernel(int m, int n, T alpha, const T* A, T* B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < n && row < m) {
        T sum = alpha * B[row * n + col];
        
        // 分块处理
        for (int block = 0; block < m; block += BLOCK_SIZE) {
            int block_end = min(block + BLOCK_SIZE, m);
            
            // 使用共享内存存储A和B的块
            __shared__ T shared_A_block[BLOCK_SIZE][BLOCK_SIZE];
            __shared__ T shared_B_block[BLOCK_SIZE];
            
            int tid_x = threadIdx.x;
            int tid_y = threadIdx.y;
            
            // 协作加载A的块
            if (block + tid_y < m && block + tid_x < m && tid_y < BLOCK_SIZE && tid_x < BLOCK_SIZE) {
                shared_A_block[tid_y][tid_x] = A[(block + tid_y) * m + block + tid_x];
            } else if (tid_y < BLOCK_SIZE && tid_x < BLOCK_SIZE) {
                shared_A_block[tid_y][tid_x] = 0;
            }
            
            // 协作加载B的块
            if (block + tid_y < m && tid_y < BLOCK_SIZE) {
                shared_B_block[tid_y] = B[(block + tid_y) * n + col];
            } else if (tid_y < BLOCK_SIZE) {
                shared_B_block[tid_y] = 0;
            }
            
            __syncthreads();
            
            // 计算部分和
            for (int k = 0; k < BLOCK_SIZE && block + k < row; ++k) {
                if (block + k < m && row >= block && row < block + BLOCK_SIZE && k < BLOCK_SIZE) {
                    sum -= shared_A_block[row - block][k] * shared_B_block[k];
                }
            }
            __syncthreads();
        }
        
        B[row * n + col] = sum / A[row * m + row];
    }
}

// 优化版本3: 向量化访问
template <typename T>
__global__ void trsm_left_lower_vectorized_kernel(int m, int n, T alpha, const T* A, T* B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < n && row < m) {
        T sum = alpha * B[row * n + col];
        
        // 向量化处理
        int vec_size = sizeof(T) == 4 ? 4 : 2;
        int vec_m = (m / vec_size) * vec_size;
        
        // 向量化部分
        for (int k = 0; k < vec_m; k += vec_size) {
            if (k + vec_size - 1 < row) {
                if constexpr (std::is_same_v<T, float>) {
                    const float4* A_vec = reinterpret_cast<const float4*>(A);
                    const float4* B_vec = reinterpret_cast<const float4*>(B);
                    int a_idx = row * m / 4 + k / 4;
                    int b_idx = k * n / 4 + col;
                    if (a_idx < (m * m / 4) && b_idx < (m * n / 4)) {
                        float4 a_val = A_vec[a_idx];
                        float4 b_val = B_vec[b_idx];
                        
                        for (int i = 0; i < 4; ++i) {
                            sum -= (&a_val.x)[i] * (&b_val.x)[i];
                        }
                    }
                } else if constexpr (std::is_same_v<T, double>) {
                    const double2* A_vec = reinterpret_cast<const double2*>(A);
                    const double2* B_vec = reinterpret_cast<const double2*>(B);
                    int a_idx = row * m / 2 + k / 2;
                    int b_idx = k * n / 2 + col;
                    if (a_idx < (m * m / 2) && b_idx < (m * n / 2)) {
                        double2 a_val = A_vec[a_idx];
                        double2 b_val = B_vec[b_idx];
                        
                        for (int i = 0; i < 2; ++i) {
                            sum -= (&a_val.x)[i] * (&b_val.x)[i];
                        }
                    }
                }
            }
        }
        
        // 处理剩余元素
        for (int k = vec_m; k < row; ++k) {
            sum -= A[row * m + k] * B[k * n + col];
        }
        
        B[row * n + col] = sum / A[row * m + row];
    }
}

template <typename T>
Trsm<T>::Trsm(int side, int uplo, int trans, int diag, T alpha)
    : side_(side), uplo_(uplo), trans_(trans), diag_(diag), alpha_(alpha) {}

template <typename T>
Trsm<T>::~Trsm() {}

template <typename T>
void Trsm<T>::SetAlpha(T alpha) {
    alpha_ = alpha;
}

template <typename T>
void Trsm<T>::SetMatrixA(const Tensor<T>& A) {
    // 由于Tensor的赋值操作符被删除，我们需要手动复制数据
    matrix_A_ = Tensor<T>(A.shape());
    cudaMemcpy(matrix_A_.data(), A.data(), A.bytes(), cudaMemcpyDeviceToDevice);
}

template <typename T>
StatusCode Trsm<T>::Forward(const Tensor<T>& input, Tensor<T>& output) {
    // 这里只实现最常用的左下三角非单位对角，非转置
    if (side_ != 0 || uplo_ != 1 || trans_ != 0 || diag_ != 0) {
        LOG(ERROR) << "Trsm: only left, lower, non-trans, non-unit supported in kernel";
        return StatusCode::UNSUPPORTED_TYPE;
    }
    int m = static_cast<int>(matrix_A_.shape()[0]);
    int n = static_cast<int>(input.shape()[1]);
    if (matrix_A_.shape()[0] != matrix_A_.shape()[1] || input.shape()[0] != m) {
        LOG(ERROR) << "Trsm: shape mismatch";
        return StatusCode::SHAPE_MISMATCH;
    }
    const T* d_A = matrix_A_.data();
    T* d_B = output.data();
    
    // 复制input到output
    cudaMemcpy(output.data(), input.data(), input.bytes(), cudaMemcpyDeviceToDevice);

    cudaError_t err = cudaSuccess;
    
    // 根据矩阵大小选择最优kernel
    if (m >= 512 && n >= 512) {
        // 大矩阵使用分块处理
        constexpr int BLOCK_SIZE = 32;
        dim3 threads(16, 16);
        dim3 blocks((n + 15) / 16, (m + 15) / 16);
        trsm_left_lower_blocked_kernel<T, BLOCK_SIZE><<<blocks, threads>>>(m, n, alpha_, d_A, d_B);
    } else if (m >= 128 && n >= 128) {
        // 中等矩阵使用并行化处理
        dim3 threads(16, 16);
        dim3 blocks((n + 15) / 16, (m + 15) / 16);
        size_t shared_mem_size = 16 * 16 * sizeof(T);
        if constexpr (std::is_same_v<T, float>) {
            trsm_left_lower_parallel_kernel_float<<<blocks, threads, shared_mem_size>>>(m, n, alpha_, d_A, d_B);
        } else if constexpr (std::is_same_v<T, double>) {
            trsm_left_lower_parallel_kernel_double<<<blocks, threads, shared_mem_size>>>(m, n, alpha_, d_A, d_B);
        }
    } else if (m >= 64 && n >= 64) {
        // 小矩阵使用向量化处理
        dim3 threads(16, 16);
        dim3 blocks((n + 15) / 16, (m + 15) / 16);
        trsm_left_lower_vectorized_kernel<T><<<blocks, threads>>>(m, n, alpha_, d_A, d_B);
    } else {
        // 极小矩阵使用基础kernel
        int threads = 32;
        int blocks = (n + threads - 1) / threads;
        trsm_left_lower_kernel<T><<<blocks, threads>>>(m, n, alpha_, d_A, d_B);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "trsm kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    
    VLOG(1) << "Trsm: solve AX = alpha*B, m = " << m << ", n = " << n;
    return StatusCode::SUCCESS;
}

// 显式实例化
template class Trsm<float>;
template class Trsm<double>;

} // namespace cu_op_mem