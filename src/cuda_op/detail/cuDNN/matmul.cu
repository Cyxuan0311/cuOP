#include "cuda_op/detail/cuDNN/matmul.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>

namespace cu_op_mem {

template <typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C, int M, int N, int K, bool transA, bool transB) {
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
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        matmul_kernel<T><<<grid, block>>>(A.data(), B.data(), C.data(), M, N, K, transA_, transB_);
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
            dim3 block(16, 16);
            dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
            matmul_kernel<T><<<grid, block>>>(
                a_ptr + b * M * K,
                b_ptr + b * K * N,
                c_ptr + b * M * N,
                M, N, K, transA_, transB_);
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
