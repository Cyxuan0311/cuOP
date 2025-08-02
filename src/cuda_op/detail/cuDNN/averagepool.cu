#include "cuda_op/detail/cuDNN/averagepool2D.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>

namespace cu_op_mem {

template <typename T>
__global__ void averagepool2D_kernel(const T* input, T* output, int input_height, int input_width, int output_height,
                                     int output_width, int pool_height, int pool_width, int stride_height,
                                     int stride_width) {
    // 共享内存块大小配置 (32x32)
    constexpr int TILE_DIM   = 32;
    constexpr int BLOCK_ROWS = 8;
    __shared__ T shared_block[TILE_DIM][TILE_DIM];

    const int output_x = blockIdx.x * TILE_DIM + threadIdx.x;
    const int output_y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (output_x >= output_width || output_y >= output_height) return;

    const int input_x_start = output_x * stride_width;
    const int input_y_start = output_y * stride_height;
    const int input_x_end = min(input_x_start + pool_width, input_width);
    const int input_y_end = min(input_y_start + pool_height, input_height);

    T sum = 0;
    int count = 0;
    for (int y = input_y_start; y < input_y_end; y += BLOCK_ROWS) {
        for (int x = input_x_start; x < input_x_end; x += TILE_DIM) {
            const int load_x = x + threadIdx.x;
            const int load_y = y + threadIdx.y;
            if (load_x < input_width && load_y < input_height) {
                shared_block[threadIdx.y][threadIdx.x] = input[load_y * input_width + load_x];
            } else {
                shared_block[threadIdx.y][threadIdx.x] = 0;
            }
            __syncthreads();
            const int search_height = min(BLOCK_ROWS, input_y_end - y);
            const int search_width  = min(TILE_DIM, input_x_end - x);
            for (int i = 0; i < search_height; ++i) {
                for (int j = 0; j < search_width; ++j) {
                    sum += shared_block[i][j];
                    ++count;
                }
            }
            __syncthreads();
        }
    }
    output[output_y * output_width + output_x] = sum / (T)count;
}

template <typename T>
StatusCode AveragePool2D<T>::Forward(const Tensor<T>& input, Tensor<T>& output, int dim_h, int dim_w) {
    const auto& input_shape = input.shape();
    if (input_shape.size() == 2) {
        // 原二维实现
        int input_height = input_shape[0];
        int input_width = input_shape[1];
        int output_height = (input_height - pool_height_) / stride_height_ + 1;
        int output_width = (input_width - pool_width_) / stride_width_ + 1;
        std::vector<std::size_t> output_shape = {static_cast<std::size_t>(output_height), static_cast<std::size_t>(output_width)};
        output.resize(output_shape);
        dim3 block_size(16, 16);
        dim3 grid_size((output_width + block_size.x - 1) / block_size.x, (output_height + block_size.y - 1) / block_size.y);
        averagepool2D_kernel<T><<<grid_size, block_size>>>(input.data(), output.data(), input_height, input_width, output_height, output_width, pool_height_, pool_width_, stride_height_, stride_width_);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "AveragePool2D kernel failed: " << cudaGetErrorString(err);
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
        int output_height = (H - pool_height_) / stride_height_ + 1;
        int output_width = (W - pool_width_) / stride_width_ + 1;
        std::vector<std::size_t> output_shape = {static_cast<std::size_t>(N), static_cast<std::size_t>(C), static_cast<std::size_t>(output_height), static_cast<std::size_t>(output_width)};
        output.resize(output_shape);
        int batch = N * C;
        const T* input_ptr = input.data();
        T* output_ptr = output.data();
        for (int i = 0; i < batch; ++i) {
            const T* in = input_ptr + i * H * W;
            T* out = output_ptr + i * output_height * output_width;
            dim3 block_size(16, 16);
            dim3 grid_size((output_width + block_size.x - 1) / block_size.x, (output_height + block_size.y - 1) / block_size.y);
            averagepool2D_kernel<T><<<grid_size, block_size>>>(in, out, H, W, output_height, output_width, pool_height_, pool_width_, stride_height_, stride_width_);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "AveragePool2D kernel failed: " << cudaGetErrorString(err);
            return StatusCode::CUDA_ERROR;
        }
        cudaDeviceSynchronize();
        return StatusCode::SUCCESS;
    } else {
        LOG(ERROR) << "AveragePool2D only supports 2D or 4D input, got " << input_shape.size() << "D";
        return StatusCode::TENSOR_DIMONSION_MISMATCH;
    }
}

template class AveragePool2D<float>;
template class AveragePool2D<double>;

} // namespace cu_op_mem
