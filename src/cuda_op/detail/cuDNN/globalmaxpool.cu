#include "cuda_op/detail/cuDNN/globalmaxpool2D.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <limits>

namespace cu_op_mem {

template <typename T>
__global__ void globalmaxpool2D_kernel(const T* input, T* output, int n) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = gridDim.x * blockDim.x;

    T my_max = std::numeric_limits<T>::lowest();
    while (i < n) {
        my_max = max(my_max, input[i]);
        i += gridSize;
    }
    sdata[tid] = my_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

template <typename T>
StatusCode GlobalMaxPool2D<T>::Forward(const Tensor<T>& input, Tensor<T>& output, int dim_h, int dim_w) {
    const auto& input_shape = input.shape();
    if (input_shape.size() == 2) {
        // 原二维实现
        int input_height = input_shape[0];
        int input_width = input_shape[1];
        output.resize({1});
        int threads = 256;
        globalmaxpool2D_kernel<T><<<1, threads>>>(input.data(), output.data(), input_height, input_width);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "GlobalMaxPool2D kernel failed: " << cudaGetErrorString(err);
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
            globalmaxpool2D_kernel<T><<<1, threads>>>(in, out, H, W);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "GlobalMaxPool2D kernel failed: " << cudaGetErrorString(err);
            return StatusCode::CUDA_ERROR;
        }
        cudaDeviceSynchronize();
        return StatusCode::SUCCESS;
    } else {
        LOG(ERROR) << "GlobalMaxPool2D only supports 2D or 4D input, got " << input_shape.size() << "D";
        return StatusCode::TENSOR_DIMONSION_MISMATCH;
    }
}

template class GlobalMaxPool2D<float>;
template class GlobalMaxPool2D<double>;

} // namespace cu_op_mem