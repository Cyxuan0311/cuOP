#include "cuda_op/detail/cuDNN/globalaverpool.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>

namespace cu_op_mem {

template <typename T>
__global__ void global_averagepool2D_kernel(const T* input, T* output, int input_height, int input_width) {
    __shared__ T shared_sum[256];
    int tid = threadIdx.x;
    int total = input_height * input_width;
    T sum = 0;
    for (int i = tid; i < total; i += blockDim.x) {
        sum += input[i];
    }
    shared_sum[tid] = sum;
    __syncthreads();
    // 归约求和
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[0] = shared_sum[0] / total;
    }
}

template <typename T>
StatusCode GlobalAveragePool2D<T>::Forward(const Tensor<T>& input, Tensor<T>& output, int dim_h, int dim_w) {
    const auto& input_shape = input.shape();
    if (input_shape.size() == 2) {
        // 原二维实现
        int input_height = input_shape[0];
        int input_width = input_shape[1];
        output.resize({1});
        int threads = 256;
        global_averagepool2D_kernel<T><<<1, threads>>>(input.data(), output.data(), input_height, input_width);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "GlobalAveragePool2D kernel failed: " << cudaGetErrorString(err);
            return StatusCode::CUDA_ERROR;
        }
        cudaDeviceSynchronize();
        return StatusCode::SUCCESS;
    } else if (input_shape.size() == 4) {
        // 四维张量 [N, C, H, W]
        int N = input_shape[0];
        int C = input_shape[1];
        int H = input_shape[2];
        int W = input_shape[3];
        std::vector<std::size_t> output_shape = {static_cast<std::size_t>(N), static_cast<std::size_t>(C), 1};
        output.resize(output_shape);
        int batch = N * C;
        const T* input_ptr = input.data();
        T* output_ptr = output.data();
        for (int i = 0; i < batch; ++i) {
            const T* in = input_ptr + i * H * W;
            T* out = output_ptr + i;
            int threads = 256;
            global_averagepool2D_kernel<T><<<1, threads>>>(in, out, H, W);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "GlobalAveragePool2D kernel failed: " << cudaGetErrorString(err);
            return StatusCode::CUDA_ERROR;
        }
        cudaDeviceSynchronize();
        return StatusCode::SUCCESS;
    } else {
        LOG(ERROR) << "GlobalAveragePool2D only supports 2D or 4D input, got " << input_shape.size() << "D";
        return StatusCode::TENSOR_DIMONSION_MISMATCH;
    }
}

template class GlobalAveragePool2D<float>;
template class GlobalAveragePool2D<double>;

} // namespace cu_op_mem
