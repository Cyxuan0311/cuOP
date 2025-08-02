#include "cuda_op/detail/cuBlas/gemm.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace cu_op_mem {

// tile优化版kernel（只支持不转置，转置可扩展）
template <typename T, int TILE>
__global__ void gemm_kernel_tiled(int m, int n, int k, T alpha, const T* A, const T* B, T beta, T* C) {
    __shared__ T As[TILE][TILE];
    __shared__ T Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    T sum = 0;

    for (int t = 0; t < (k + TILE - 1) / TILE; ++t) {
        int tiled_col = t * TILE + threadIdx.x;
        int tiled_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < m && tiled_col < k) ? A[row * k + tiled_col] : 0;
        Bs[threadIdx.y][threadIdx.x] = (col < n && tiled_row < k) ? B[tiled_row * n + col] : 0;
        __syncthreads();

        for (int i = 0; i < TILE; ++i)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }

    if (row < m && col < n)
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
}

// 兼容原有接口，自动选择kernel
template <typename T>
Gemm<T>::Gemm(bool transA, bool transB, T alpha, T beta)
    : transA_(transA), transB_(transB), alpha_(alpha), beta_(beta) {}

template <typename T>
Gemm<T>::~Gemm() {}

template <typename T>
void Gemm<T>::SetWeight(const Tensor<T>& weight) {
    weight_ = Tensor<T>(weight.shape());
    cudaMemcpy(weight_.data(), weight.data(), weight.bytes(), cudaMemcpyDeviceToDevice);
}

template <typename T>
StatusCode Gemm<T>::Forward(const Tensor<T>& input, Tensor<T>& output) {
    int m = static_cast<int>(input.shape()[0]);
    int k = static_cast<int>(input.shape()[1]);
    int n = static_cast<int>(weight_.shape()[1]);
    if (weight_.shape()[0] != k || output.shape()[0] != m || output.shape()[1] != n) {
        LOG(ERROR) << "Gemm: shape mismatch";
        return StatusCode::SHAPE_MISMATCH;
    }
    const T* d_A = input.data();
    const T* d_B = weight_.data();
    T* d_C = output.data();

    constexpr int TILE = 16;
    dim3 threads(TILE, TILE);
    dim3 blocks((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);

    // 只优化不转置的情况，转置时仍用原kernel
    if (!transA_ && !transB_) {
        gemm_kernel_tiled<T, TILE><<<blocks, threads>>>(m, n, k, alpha_, d_A, d_B, beta_, d_C);
    } else {
        // 原始kernel支持转置
        __global__ void gemm_kernel(int m, int n, int k, T alpha, const T* A, const T* B, T beta, T* C, bool transA, bool transB);
        dim3 threads2(16, 16);
        dim3 blocks2((n + 15) / 16, (m + 15) / 16);
        gemm_kernel<T><<<blocks2, threads2>>>(m, n, k, alpha_, d_A, d_B, beta_, d_C, transA_, transB_);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "gemm kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    VLOG(1) << "Gemm: C = " << alpha_ << " * A * B + " << beta_ << " * C, m = " << m << ", n = " << n << ", k = " << k;
    return StatusCode::SUCCESS;
}

// 显式实例化
template class Gemm<float>;
template class Gemm<double>;

} //