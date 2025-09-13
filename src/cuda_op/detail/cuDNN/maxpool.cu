#include "cuda_op/detail/cuDNN/maxpool.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>

namespace cu_op_mem {

// 优化的MaxPool2D kernel，改进共享内存使用和边界处理
template <typename T>
__global__ void maxpool2D_optimized_kernel(const T* input, T* output, int input_height, int input_width, int output_height,
                                          int output_width, int pool_height, int pool_width, int stride_height,
                                          int stride_width) {
    // 优化的块大小配置
    constexpr int TILE_DIM = 16;  // 减小块大小以减少共享内存使用

    // 共享内存缓存输入块
    __shared__ T shared_block[TILE_DIM][TILE_DIM];

    // 计算输出坐标
    const int output_x = blockIdx.x * TILE_DIM + threadIdx.x;
    const int output_y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 边界检查
    if (output_x >= output_width || output_y >= output_height) return;

    // 计算输入区域起始坐标
    const int input_x_start = output_x * stride_width;
    const int input_y_start = output_y * stride_height;

    // 计算输入区域结束坐标 (考虑边界)
    const int input_x_end = min(input_x_start + pool_width, input_width);
    const int input_y_end = min(input_y_start + pool_height, input_height);

    // 使用向量化内存访问优化
    T max_val = -INFINITY;
    
    // 直接计算最大值，避免不必要的共享内存操作
    for (int y = input_y_start; y < input_y_end; ++y) {
        for (int x = input_x_start; x < input_x_end; ++x) {
            if (x < input_width && y < input_height) {
                T val = input[y * input_width + x];
                max_val = max(max_val, val);
            }
        }
    }

    // 写入输出
    output[output_y * output_width + output_x] = max_val;
}

// 使用共享内存的MaxPool2D kernel（用于大池化窗口）
template <typename T>
__global__ void maxpool2D_shared_kernel(const T* input, T* output, int input_height, int input_width, int output_height,
                                       int output_width, int pool_height, int pool_width, int stride_height,
                                       int stride_width) {
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;

    // 共享内存缓存输入块
    __shared__ T shared_block[TILE_DIM][TILE_DIM];

    // 计算输出坐标
    const int output_x = blockIdx.x * TILE_DIM + threadIdx.x;
    const int output_y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 边界检查
    if (output_x >= output_width || output_y >= output_height) return;

    // 计算输入区域起始坐标
    const int input_x_start = output_x * stride_width;
    const int input_y_start = output_y * stride_height;

    // 计算输入区域结束坐标 (考虑边界)
    const int input_x_end = min(input_x_start + pool_width, input_width);
    const int input_y_end = min(input_y_start + pool_height, input_height);

    // 每个线程处理多个输入元素 (优化内存访问)
    T max_val = -INFINITY;
    for (int y = input_y_start; y < input_y_end; y += BLOCK_ROWS) {
        for (int x = input_x_start; x < input_x_end; x += TILE_DIM) {
            // 将输入块加载到共享内存 (协作加载)
            const int load_x = x + threadIdx.x;
            const int load_y = y + threadIdx.y;

            if (load_x < input_width && load_y < input_height) {
                shared_block[threadIdx.y][threadIdx.x] = input[load_y * input_width + load_x];
            } else {
                shared_block[threadIdx.y][threadIdx.x] = -INFINITY;
            }
            __syncthreads();

            // 在共享内存块中查找最大值
            const int search_height = min(BLOCK_ROWS, input_y_end - y);
            const int search_width = min(TILE_DIM, input_x_end - x);

            for (int i = 0; i < search_height; ++i) {
                for (int j = 0; j < search_width; ++j) {
                    max_val = max(max_val, shared_block[i][j]);
                }
            }
            __syncthreads();
        }
    }

    // 写入输出
    output[output_y * output_width + output_x] = max_val;
}

template <typename T>
StatusCode MaxPool2D<T>::Forward(const Tensor<T>& input, Tensor<T>& output, int dim_h, int dim_w) {
    const auto& input_shape = input.shape();
    if (input_shape.size() == 2) {
        // 原二维实现
        int input_height = input_shape[0];
        int input_width = input_shape[1];
        int output_height = (input_height - pool_height_) / stride_height_ + 1;
        int output_width = (input_width - pool_width_) / stride_width_ + 1;
        std::vector<std::size_t> output_shape = {static_cast<std::size_t>(output_height), static_cast<std::size_t>(output_width)};
        output.resize(output_shape);
        // 根据池化窗口大小选择合适的kernel
        if (pool_height_ * pool_width_ > 16) {
            // 大池化窗口使用共享内存kernel
            dim3 block_size(16, 16);
            dim3 grid_size((output_width + block_size.x - 1) / block_size.x, (output_height + block_size.y - 1) / block_size.y);
            maxpool2D_shared_kernel<T><<<grid_size, block_size>>>(input.data(), output.data(), input_height, input_width, output_height, output_width, pool_height_, pool_width_, stride_height_, stride_width_);
        } else {
            // 小池化窗口使用优化kernel
            dim3 block_size(16, 16);
            dim3 grid_size((output_width + block_size.x - 1) / block_size.x, (output_height + block_size.y - 1) / block_size.y);
            maxpool2D_optimized_kernel<T><<<grid_size, block_size>>>(input.data(), output.data(), input_height, input_width, output_height, output_width, pool_height_, pool_width_, stride_height_, stride_width_);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "MaxPool2D kernel failed: " << cudaGetErrorString(err);
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
            
            // 根据池化窗口大小选择合适的kernel
            if (pool_height_ * pool_width_ > 16) {
                dim3 block_size(16, 16);
                dim3 grid_size((output_width + block_size.x - 1) / block_size.x, (output_height + block_size.y - 1) / block_size.y);
                maxpool2D_shared_kernel<T><<<grid_size, block_size>>>(in, out, H, W, output_height, output_width, pool_height_, pool_width_, stride_height_, stride_width_);
            } else {
                dim3 block_size(16, 16);
                dim3 grid_size((output_width + block_size.x - 1) / block_size.x, (output_height + block_size.y - 1) / block_size.y);
                maxpool2D_optimized_kernel<T><<<grid_size, block_size>>>(in, out, H, W, output_height, output_width, pool_height_, pool_width_, stride_height_, stride_width_);
            }
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "MaxPool2D kernel failed: " << cudaGetErrorString(err);
            return StatusCode::CUDA_ERROR;
        }
        cudaDeviceSynchronize();
        return StatusCode::SUCCESS;
    } else {
        LOG(ERROR) << "MaxPool2D only supports 2D or 4D input, got " << input_shape.size() << "D";
        return StatusCode::TENSOR_DIMONSION_MISMATCH;
    }
}

template class MaxPool2D<float>;
template class MaxPool2D<double>;

} // namespace cu_op_mem