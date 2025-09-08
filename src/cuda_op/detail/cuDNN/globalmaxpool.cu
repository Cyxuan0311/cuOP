#include "cuda_op/detail/cuDNN/globalmaxpool.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <limits>

namespace cu_op_mem {

// 为float类型创建独立的kernel函数
__global__ void globalmaxpool2D_kernel_float(const float* input, float* output, int n) {
    extern __shared__ float globalmaxpool_shared_mem_float[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = gridDim.x * blockDim.x;

    float my_max = -1e30f;  // 使用一个很小的值作为初始值
    while (i < n) {
        my_max = max(my_max, input[i]);
        i += gridSize;
    }
    globalmaxpool_shared_mem_float[tid] = my_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            globalmaxpool_shared_mem_float[tid] = max(globalmaxpool_shared_mem_float[tid], globalmaxpool_shared_mem_float[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = globalmaxpool_shared_mem_float[0];
    }
}

// 为double类型创建独立的kernel函数
__global__ void globalmaxpool2D_kernel_double(const double* input, double* output, int n) {
    extern __shared__ double globalmaxpool_shared_mem_double[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = gridDim.x * blockDim.x;

    double my_max = -1e30;  // 使用一个很小的值作为初始值
    while (i < n) {
        my_max = max(my_max, input[i]);
        i += gridSize;
    }
    globalmaxpool_shared_mem_double[tid] = my_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            globalmaxpool_shared_mem_double[tid] = max(globalmaxpool_shared_mem_double[tid], globalmaxpool_shared_mem_double[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = globalmaxpool_shared_mem_double[0];
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
        
        // 根据类型调用相应的kernel函数
        if constexpr (std::is_same_v<T, float>) {
            globalmaxpool2D_kernel_float<<<1, threads>>>(input.data(), output.data(), input_height * input_width);
        } else if constexpr (std::is_same_v<T, double>) {
            globalmaxpool2D_kernel_double<<<1, threads>>>(input.data(), output.data(), input_height * input_width);
        } else {
            LOG(ERROR) << "GlobalMaxPool2D only supports float and double types.";
            return StatusCode::UNSUPPORTED_OPERATION;
        }
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
            
            // 根据类型调用相应的kernel函数
            if constexpr (std::is_same_v<T, float>) {
                globalmaxpool2D_kernel_float<<<1, threads>>>(in, out, H * W);
            } else if constexpr (std::is_same_v<T, double>) {
                globalmaxpool2D_kernel_double<<<1, threads>>>(in, out, H * W);
            } else {
                LOG(ERROR) << "GlobalMaxPool2D only supports float and double types.";
                return StatusCode::UNSUPPORTED_OPERATION;
            }
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