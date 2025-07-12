#include "cuda_op/detail/cuBlas/gemv.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace cu_op_mem {

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

    int threads = 256;
    int blocks = (m + threads - 1) / threads;
    gemv_kernel<T><<<blocks, threads>>>(m, n, alpha_, d_A, d_x, beta_, d_y, transA_);

    cudaError_t err = cudaGetLastError();
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