#include "cuda_op/detail/cuDNN/batchmatmul.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>

namespace cu_op_mem {

constexpr int TILE = 16;

template <typename T, int TILE_SIZE>
__global__ void batchmatmul_shared_kernel(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    int batch_stride_A, int batch_stride_B, int batch_stride_C,
    int batch)
{
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    const T* A_ptr = A + batch_idx * batch_stride_A;
    const T* B_ptr = B + batch_idx * batch_stride_B;
    T* C_ptr = C + batch_idx * batch_stride_C;

    T sum = 0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        __shared__ T Asub[TILE_SIZE][TILE_SIZE];
        __shared__ T Bsub[TILE_SIZE][TILE_SIZE];

        int tiledRow = row;
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        Asub[threadIdx.y][threadIdx.x] = (tiledRow < M && tiledCol < K) ? A_ptr[tiledRow * K + tiledCol] : 0;

        tiledRow = t * TILE_SIZE + threadIdx.y;
        tiledCol = col;
        Bsub[threadIdx.y][threadIdx.x] = (tiledRow < K && tiledCol < N) ? B_ptr[tiledRow * N + tiledCol] : 0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C_ptr[row * N + col] = sum;
    }
}

template <typename T>
StatusCode BatchMatMul<T>::Forward(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C, int batch_dim) {
    const auto& a_shape = A.shape();
    const auto& b_shape = B.shape();
    if (a_shape.size() < 3 || b_shape.size() < 3) {
        LOG(ERROR) << "BatchMatMul: input tensors must be at least 3D";
        return StatusCode::SHAPE_MISMATCH;
    }
    if (batch_dim < 0) batch_dim += a_shape.size();
    if (batch_dim < 0 || batch_dim >= (int)a_shape.size()) {
        LOG(ERROR) << "BatchMatMul: batch_dim out of range";
        return StatusCode::SHAPE_MISMATCH;
    }
    // 假设 batch 维度一致
    int batch = 1;
    for (int i = 0; i <= batch_dim; ++i) batch *= a_shape[i];
    int M = a_shape[batch_dim + 1];
    int K = a_shape[batch_dim + 2];
    int N = b_shape[batch_dim + 2];
    int batch_stride_A = M * K;
    int batch_stride_B = K * N;
    int batch_stride_C = M * N;
    // 输出 shape
    std::vector<std::size_t> c_shape = a_shape;
    c_shape[batch_dim + 2] = N;
    if (C.data() == nullptr || C.shape() != c_shape) {
        C = Tensor<T>(c_shape);
    }
    dim3 block(TILE, TILE, 1);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch);
    batchmatmul_shared_kernel<T, TILE><<<grid, block>>>(
        A.data(), B.data(), C.data(),
        M, N, K,
        batch_stride_A, batch_stride_B, batch_stride_C,
        batch
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "BatchMatMul kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    cudaDeviceSynchronize();
    return StatusCode::SUCCESS;
}

// 显式实例化
template class BatchMatMul<float>;
template class BatchMatMul<double>;

} // namespace cu_op_mem
