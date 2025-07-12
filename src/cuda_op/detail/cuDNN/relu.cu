#include "cuda_op/detail/cuDNN/relu.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>

namespace cu_op_mem {

template <typename T>
__global__ void relu_kernel(const T* in, T* out, std::size_t N) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        T v      = in[idx];
        out[idx] = v > T(0) ? v : T(0);
    }
}

template <typename T>
StatusCode Relu<T>::Forward(const Tensor<T>& input, Tensor<T>& output) {
    std::size_t N = input.numel();

    if (output.data() == nullptr) {
        output = Tensor<T>(input.shape());
        // output = std::make_shared<Tensor<T>>(input->shape());
    } else if (output.numel() != N) {
        throw std::runtime_error("Relu's output and input tensor not match");
    }

    const T* d_in  = input.data();
    T*       d_out = output.data();

    const std::size_t threads = 256;
    const std::size_t blocks  = (N + threads - 1) / threads;
    LOG(INFO) << "Relu launch: blocks = " << blocks << " , threads = " << threads << " , N = " << N;
    relu_kernel<T><<<blocks, threads>>>(d_in, d_out, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "Relu kernel failed: " << cudaGetErrorString(err);
        throw std::runtime_error("Relu kernel failed");
    }

    return StatusCode::SUCCESS;
}

template class Relu<float>;
template class Relu<double>;

} // namespace cu_op_mem