#include "cuda_op/detail/cuBlas/axpy.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace cu_op_mem {

template <typename T>
__global__ void axpy_kernel(int n, T alpha, const T* x, T* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

template <typename T>
Axpy<T>::Axpy(T alpha) : alpha_(alpha) {}

template <typename T>
Axpy<T>::~Axpy() {}

template <typename T>
void Axpy<T>::SetAlpha(T alpha) {
    alpha_ = alpha;
}

template <typename T>
StatusCode Axpy<T>::Forward(const Tensor<T>& x, Tensor<T>& y) {
    if (x.numel() != y.numel()) {
        LOG(ERROR) << "Axpy: x and y must have the same number of elements";
        return StatusCode::SHAPE_MISMATCH;
    }
    int n = static_cast<int>(x.numel());
    const T* d_x = x.data();
    T* d_y = y.data();

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    axpy_kernel<T><<<blocks, threads>>>(n, alpha_, d_x, d_y);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "axpy kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    VLOG(1) << "Axpy: y = " << alpha_ << " * x + y, n = " << n;
    return StatusCode::SUCCESS;
}

// 显式实例化
template class Axpy<float>;
template class Axpy<double>;

} // namespace cu_op_mem