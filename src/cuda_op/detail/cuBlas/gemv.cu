#include <glog/logging.h>
#include <stdexcept>
#include "cuda_op/detail/cuBlas/gemv.hpp"

namespace cu_op_mem {

template <typename T>
Gemv<T>::Gemv(bool transA, T alpha, T beta)
    : transA_(transA), alpha_(alpha), beta_(beta) {
    cublasStatus_t status = cublasCreate(&handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "Failed to create cuBLAS handle, status: " << status;
        throw std::runtime_error("Failed to create cuBLAS handle");
    }
}

template <typename T>
Gemv<T>::~Gemv() {
    cublasDestroy(handle_);
}

template <typename T>
void Gemv<T>::SetWeight(const Tensor<T>& weight) {
    weight_ = Tensor<T>(weight.shape());
    cudaMemcpy(weight_.data(), weight.data(), weight.bytes(), cudaMemcpyDeviceToDevice);
}

template <typename T>
void Gemv<T>::Forward(const Tensor<T>& input, Tensor<T>& output) {
    // input: (M, N), weight: (N,), output: (M,)
    int m = input.shape(0);
    int n = input.shape(1);

    cublasOperation_t opA = transA_ ? CUBLAS_OP_T : CUBLAS_OP_N;
    const T* A = input.data();
    const T* x = weight_.data();
    T* y = output.data();

    int lda = transA_ ? m : n;
    int incx = 1;
    int incy = 1;

    cublasStatus_t status;
    if constexpr (std::is_same<T, float>::value) {
        status = cublasSgemv(handle_, opA, m, n, &alpha_, A, lda, x, incx, &beta_, y, incy);
    } else if constexpr (std::is_same<T, double>::value) {
        status = cublasDgemv(handle_, opA, m, n, &alpha_, A, lda, x, incx, &beta_, y, incy);
    } else {
        LOG(ERROR) << "Unsupported data type for Gemv.";
        throw std::runtime_error("Unsupported data type for Gemv");
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "cuBLAS gemv failed, status: " << status;
        throw std::runtime_error("cuBLAS gemv failed");
    }
}

// 显式实例化
template class Gemv<float>;
template class Gemv<double>;

} // namespace cu_op_mem
