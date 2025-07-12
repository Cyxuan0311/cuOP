#include "cuda_op/detail/cuBlas/gemm.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace cu_op_mem {

template <typename T>
__global__ void gemm_kernel(int m, int n, int k, T alpha, const T* A, const T* B, T beta, T* C, bool transA, bool transB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        T sum = 0;
        for (int i = 0; i < k; ++i) {
            T a = transA ? A[i * m + row] : A[row * k + i];
            T b = transB ? B[col * k + i] : B[i * n + col];
            sum += a * b;
        }
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}

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

    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (m + 15) / 16);
    gemm_kernel<T><<<blocks, threads>>>(m, n, k, alpha_, d_A, d_B, beta_, d_C, transA_, transB_);

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

} // namespace cu_op_mem