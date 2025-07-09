#include "cuda_op/detail/cuBlas/symm.hpp"
#include <glog/logging.h>
#include <stdexcept>

namespace cu_op_mem {

template <typename T>
Symm<T>::Symm(cublasSideMode_t side, cublasFillMode_t uplo, T alpha, T beta)
    : side_(side), uplo_(uplo), alpha_(alpha), beta_(beta) {
    cublasStatus_t status = cublasCreate(&handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        LOG(FATAL) << "Failed to create cuBLAS handle!";
    }
}

template <typename T>
Symm<T>::~Symm() {
    cublasStatus_t status = cublasDestroy(handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "Failed to destroy cuBLAS handle!";
    }
}

template <typename T>
void Symm<T>::SetWeight(const Tensor<T>& weight) {
    weight_ = Tensor<T>(weight.shape());
    cudaMemcpy(weight_.data(), weight.data(), weight.bytes(), cudaMemcpyDeviceToDevice);
    VLOG(1) << "Symm weight set, shape=[" << weight.shape()[0] << ", " << weight.shape()[1] << "]";
}

template <typename T>
void Symm<T>::Forward(const Tensor<T>& input, Tensor<T>& output) {
    // input: (M, N), weight: (M, M) or (N, N), output: (M, N)
    if (input.shape().size() != 2 || weight_.shape().size() != 2) {
        LOG(ERROR) << "Symm only supports 2D tensors";
        throw std::invalid_argument("Symm only supports 2D tensors");
    }
    int m = static_cast<int>(input.shape()[0]);
    int n = static_cast<int>(input.shape()[1]);
    int lda = (side_ == CUBLAS_SIDE_LEFT) ? m : n;
    int ldb = n;
    int ldc = n;
    const T* A = weight_.data();
    const T* B = input.data();
    T* C = output.data();
    VLOG(1) << "Symm launch: m=" << m << ", n=" << n << ", lda=" << lda << ", ldb=" << ldb << ", ldc=" << ldc;
    cublasStatus_t status;
    if constexpr (std::is_same<T, float>::value) {
        status = cublasSsymm(handle_, side_, uplo_, m, n, &alpha_, A, lda, B, ldb, &beta_, C, ldc);
    } else if constexpr (std::is_same<T, double>::value) {
        status = cublasDsymm(handle_, side_, uplo_, m, n, &alpha_, A, lda, B, ldb, &beta_, C, ldc);
    } else {
        LOG(ERROR) << "Unsupported data type for Symm.";
        throw std::runtime_error("Unsupported data type for Symm");
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "cuBLAS symm failed, status: " << status;
        throw std::runtime_error("cuBLAS symm failed");
    }
}

// 显式实例化
template class Symm<float>;
template class Symm<double>;

} // namespace cu_op_mem
