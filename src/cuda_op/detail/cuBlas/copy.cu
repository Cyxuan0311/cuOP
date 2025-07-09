#include "cuda_op/detail/cuBlas/copy.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace cu_op_mem {
    template <typename T>
    __global__ void copy_kernel(int n,const T* x,T* y){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < n) {
            y[idx] = x[idx];
        }
    }

    template <typename T>
    Copy<T>::Copy() {}

    template <typename T>
    Copy<T>::~Copy() {}

    template <typename T>
    StatusCode Copy<T>::Forward(const Tensor<T>& x,Tensor<T>& y) {
        if(x.numel() != y.numel()) {
            LOG(ERROR)<<"Copy: x and y must have the same number of elements";
            return StatusCode::SHAPE_MISMATCH;
        }
        int n = static_cast<int>(x.numel());
        const T* d_x = x.data();
        T* d_y = y.data();

        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        copy_kernel<T><<<blocks,threads>>>>(n,d_x,d_y);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "copy kernel failed: "<< cudaGetErrorString(err);
            return StatusCode::CUDA_ERROR;
        }
        VLOG(1)<<"Copy: y = x, n = "<<n;
        return StatusCode::SUCCESS;
    }

    template class Copy<float>;
    template class Copy<double>;
    
}