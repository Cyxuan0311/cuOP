#include "cuda_op/detail/cuBlas/gemm.hpp"
#include <cublas_v2.h>
#include <stdexcept>
#include <glog/logging.h>

namespace cu_op_mem {

template <typename T>
Gemm<T>::Gemm(bool transA, bool transB, T alpha, T beta)
    : transA_(transA), transB_(transB), alpha_(alpha), beta_(beta) {
    cublasStatus_t status = cublasCreate(&handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        LOG(FATAL) << "Failed to create cuBLAS handle!";
    }
}

template <typename T>
Gemm<T>::~Gemm() {
    cublasStatus_t status = cublasDestroy(handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "Failed to destroy cuBLAS handle!";
    }
}

template <typename T>
void Gemm<T>::SetWeight(const Tensor<T>& weight) {
    weight_ = Tensor<T>(weight.shape());
    cudaMemcpy(weight_.data(), weight.data(), weight.bytes(), cudaMemcpyDeviceToDevice);
}

template <>
void Gemm<float>::Forward(const Tensor<float>& input, Tensor<float>& output) {
    int m = static_cast<int>(input.shape()[0]);
    int k = static_cast<int>(input.shape()[1]);
    int n = static_cast<int>(weight_.shape()[1]);

    const float* A = input.data();
    const float* B = weight_.data();
    float* C = output.data();

    cublasOperation_t opA = transA_ ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB_ ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasStatus_t status = cublasSgemm(
        handle_,
        opB, opA,
        n, m, k,
        &alpha_,
        B, transB_ ? k : n,
        A, transA_ ? m : k,
        &beta_,
        C, n
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        LOG(FATAL) << "cublasSgemm failed with status: " << status;
    }
}

template class Gemm<float>;

} // namespace cu_op_mem