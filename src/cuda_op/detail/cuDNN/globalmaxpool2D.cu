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
StatusCode GlobalMaxPool2D<T>::Forward(const Tensor<T>& input, Tensor<T>& output) {
    const auto& input_shape = input.shape();
    if (input_shape.size() != 2) {
        LOG(ERROR) << "Input must be 2D tensor";
        return StatusCode::TENSOR_DIMONSION_MISMATCH;
    }
    int input_height = input_shape[0];
    int input_width = input_shape[1];
    int n = input_height * input_width;

    if (n == 0) {
        LOG(WARNING) << "Input tensor is empty";
        output.resize({1});
        T val = std::numeric_limits<T>::lowest();
        cudaMemcpy(output.data(), &val, sizeof(T), cudaMemcpyHostToDevice);
        return StatusCode::SUCCESS;
    }

    output.resize({1});

    int threads = 512;
    int blocks = (n + threads - 1) / threads;
    LOG(INFO) << "GlobalMaxPool2D launch: n=" << n << ", blocks=" << blocks << ", threads=" << threads;
    
    if (blocks > 1) {
        Tensor<T> partial_max_tensor;
        partial_max_tensor.resize({(size_t)blocks});

        globalmaxpool2D_kernel<T><<<blocks, threads, threads * sizeof(T)>>>(
            input.data(), partial_max_tensor.data(), n);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "GlobalMaxPool2D kernel (pass 1) failed: " << cudaGetErrorString(err);
            return StatusCode::CUDA_ERROR;
        }

        LOG(INFO) << "GlobalMaxPool2D launch (pass 2): n=" << blocks << ", blocks=1, threads=" << threads;
        globalmaxpool2D_kernel<T><<<1, threads, threads * sizeof(T)>>>(
            partial_max_tensor.data(), output.data(), blocks);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "GlobalMaxPool2D kernel (pass 2) failed: " << cudaGetErrorString(err);
            return StatusCode::CUDA_ERROR;
        }
    } else {
        globalmaxpool2D_kernel<T><<<1, threads, threads * sizeof(T)>>>(
            input.data(), output.data(), n);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "GlobalMaxPool2D kernel failed: " << cudaGetErrorString(err);
            return StatusCode::CUDA_ERROR;
        }
    }

    cudaDeviceSynchronize();
    return StatusCode::SUCCESS;
}

template class GlobalMaxPool2D<float>;
template class GlobalMaxPool2D<double>;

} // namespace cu_op_mem