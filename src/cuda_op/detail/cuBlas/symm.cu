#include "cuda_op/detail/cuBlas/symm.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace cu_op_mem {

template <typename T>
__global__ void symm_kernel(int m, int n, T alpha, const T* A, const T* B, T beta, T* C, bool left, bool upper) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        T sum = 0;
        if (left) {
            for (int k = 0; k < m; ++k) {
                T a = upper ? (row <= k ? A[row * m + k] : A[k * m + row]) : (row >= k ? A[row * m + k] : A[k * m + row]);
                sum += a * B[k * n + col];
            }
        } else {
            for (int k = 0; k < n; ++k) {
                T a = upper ? (col <= k ? A[col * n + k] : A[k * n + col]) : (col >= k ? A[col * n + k] : A[k * n + col]);
                sum += B[row * n + k] * a;
            }
        }
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}

template <typename T>
Symm<T>::Symm(cublasSideMode_t side, cublasFillMode_t uplo, T alpha, T beta)
    : side_(side), uplo_(uplo), alpha_(alpha), beta_(beta) {}

template <typename T>
Symm<T>::~Symm() {}

template <typename T>
void Symm<T>::SetWeight(const Tensor<T>& weight) {
    weight_ = Tensor<T>(weight.shape());
    cudaMemcpy(weight_.data(), weight.data(), weight.bytes(), cudaMemcpyDeviceToDevice);
    VLOG(1) << "Symm weight set, shape=[" << weight.shape()[0] << ", " << weight.shape()[1] << "]";
}

template <typename T>
StatusCode Symm<T>::Forward(const Tensor<T>& input, Tensor<T>& output) {
    int m = static_cast<int>(input.shape()[0]);
    int n = static_cast<int>(input.shape()[1]);
    if (weight_.shape()[0] != m || weight_.shape()[1] != m || output.shape()[0] != m || output.shape()[1] != n) {
        LOG(ERROR) << "Symm: shape mismatch";
        return StatusCode::SHAPE_MISMATCH;
    }
    const T* d_A = weight_.data();
    const T* d_B = input.data();
    T* d_C = output.data();

    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (m + 15) / 16);
    bool left = (side_ == 0); // CUBLAS_SIDE_LEFT
    bool upper = (uplo_ == 0); // CUBLAS_FILL_MODE_UPPER
    symm_kernel<T><<<blocks, threads>>>(m, n, alpha_, d_A, d_B, beta_, d_C, left, upper);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "symm kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    VLOG(1) << "Symm: C = " << alpha_ << " * A * B + " << beta_ << " * C, m = " << m << ", n = " << n;
    return StatusCode::SUCCESS;
}

// 显式实例化
template class Symm<float>;
template class Symm<double>;

} // namespace cu_op_mem