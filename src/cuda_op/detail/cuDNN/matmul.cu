#include "cuda_op/detail/cuDNN/matmul.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>

namespace cu_op_mem {

// 优化的矩阵乘法kernel，使用共享内存和分块技术
template <typename T>
__global__ void matmul_optimized_kernel(const T* A, const T* B, T* C, int M, int N, int K, bool transA, bool transB) {
    // 分块大小
    const int BLOCK_SIZE = 16;
    
    // 共享内存用于缓存数据块
    __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    T sum = 0;
    
    // 分块矩阵乘法
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // 协作加载A块到共享内存
        int A_row = row;
        int A_col = tile * BLOCK_SIZE + threadIdx.x;
        if (A_row < M && A_col < K) {
            As[threadIdx.y][threadIdx.x] = transA ? A[A_col * M + A_row] : A[A_row * K + A_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }
        
        // 协作加载B块到共享内存
        int B_row = tile * BLOCK_SIZE + threadIdx.y;
        int B_col = col;
        if (B_row < K && B_col < N) {
            Bs[threadIdx.y][threadIdx.x] = transB ? B[B_col * K + B_row] : B[B_row * N + B_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // 计算部分积
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // 写入结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 简单的矩阵乘法kernel（用于小矩阵）
template <typename T>
__global__ void matmul_simple_kernel(const T* A, const T* B, T* C, int M, int N, int K, bool transA, bool transB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; ++k) {
            T a = transA ? A[k * M + row] : A[row * K + k];
            T b = transB ? B[col * K + k] : B[k * N + col];
            sum += a * b;
        }
        C[row * N + col] = sum;
    }
}

template <typename T>
StatusCode MatMul<T>::Forward(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C, int batch_dim) {
    const auto& a_shape = A.shape();
    const auto& b_shape = B.shape();
    // 自动推断 batch 维度
    int a_ndim = a_shape.size();
    int b_ndim = b_shape.size();
    if (a_ndim == 2 && b_ndim == 2) {
        // 原二维实现
        int M = transA_ ? a_shape[1] : a_shape[0];
        int K = transA_ ? a_shape[0] : a_shape[1];
        int N = transB_ ? b_shape[0] : b_shape[1];
        int Kb = transB_ ? b_shape[1] : b_shape[0];
        if (K != Kb) {
            LOG(ERROR) << "Inner dimensions do not match for matmul";
            return StatusCode::SHAPE_MISMATCH;
        }
        C.resize({static_cast<size_t>(M), static_cast<size_t>(N)});
        
        // 根据矩阵大小选择kernel
        if (M >= 32 && N >= 32 && K >= 32) {
            // 大矩阵使用优化的分块kernel
            dim3 block(16, 16);
            dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
            matmul_optimized_kernel<T><<<grid, block>>>(A.data(), B.data(), C.data(), M, N, K, transA_, transB_);
        } else {
            // 小矩阵使用简单kernel
            dim3 block(16, 16);
            dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
            matmul_simple_kernel<T><<<grid, block>>>(A.data(), B.data(), C.data(), M, N, K, transA_, transB_);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "MatMul kernel failed: " << cudaGetErrorString(err);
            return StatusCode::CUDA_ERROR;
        }
        cudaDeviceSynchronize();
        return StatusCode::SUCCESS;
    } else if (a_ndim == 3 && b_ndim == 3) {
        // batch matmul: [B, M, K] x [B, K, N] -> [B, M, N]
        int BATCH = a_shape[0];
        int M = a_shape[1];
        int K = a_shape[2];
        int Kb = b_shape[1];
        int N = b_shape[2];
        if (BATCH != b_shape[0] || K != Kb) {
            LOG(ERROR) << "Batch or inner dimensions do not match for batch matmul";
            return StatusCode::SHAPE_MISMATCH;
        }
        C.resize({static_cast<size_t>(BATCH), static_cast<size_t>(M), static_cast<size_t>(N)});
        const T* a_ptr = A.data();
        const T* b_ptr = B.data();
        T* c_ptr = C.data();
        for (int b = 0; b < BATCH; ++b) {
            // 根据矩阵大小选择kernel
            if (M >= 32 && N >= 32 && K >= 32) {
                dim3 block(16, 16);
                dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
                matmul_optimized_kernel<T><<<grid, block>>>(
                    a_ptr + b * M * K,
                    b_ptr + b * K * N,
                    c_ptr + b * M * N,
                    M, N, K, transA_, transB_);
            } else {
                dim3 block(16, 16);
                dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
                matmul_simple_kernel<T><<<grid, block>>>(
                    a_ptr + b * M * K,
                    b_ptr + b * K * N,
                    c_ptr + b * M * N,
                    M, N, K, transA_, transB_);
            }
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "Batch MatMul kernel failed: " << cudaGetErrorString(err);
            return StatusCode::CUDA_ERROR;
        }
        cudaDeviceSynchronize();
        return StatusCode::SUCCESS;
    } else {
        LOG(ERROR) << "MatMul only supports 2D or 3D (batch) tensors, got " << a_ndim << "D and " << b_ndim << "D";
        return StatusCode::TENSOR_DIMONSION_MISMATCH;
    }
}

template class MatMul<float>;
template class MatMul<double>;

} // namespace cu_op_mem
