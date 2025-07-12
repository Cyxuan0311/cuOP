#include "cuda_op/detail/cuBlas/trsm.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace cu_op_mem {

// 仅支持 side=left, uplo=lower, trans=no, diag=non-unit，单线程实现
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
StatusCode Trsm<T>::Forward(const Tensor<T>& A, Tensor<T>& B) {
    // 这里只实现最常用的左下三角非单位对角，非转置
    if (side_ != 0 || uplo_ != 1 || trans_ != 0 || diag_ != 0) {
        LOG(ERROR) << "Trsm: only left, lower, non-trans, non-unit supported in kernel";
        return StatusCode::UNSUPPORTED_TYPE;
    }
    int m = static_cast<int>(A.shape()[0]);
    int n = static_cast<int>(B.shape()[1]);
    if (A.shape()[0] != A.shape()[1] || B.shape()[0] != m) {
        LOG(ERROR) << "Trsm: shape mismatch";
        return StatusCode::SHAPE_MISMATCH;
    }
    const T* d_A = A.data();
    T* d_B = B.data();

    int threads = 32;
    int blocks = (n + threads - 1) / threads;
    trsm_left_lower_kernel<T><<<blocks, threads>>>(m, n, alpha_, d_A, d_B);

    cudaError_t err = cudaGetLastError();
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