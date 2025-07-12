#include "cuda_op/detail/cuBlas/scal.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace cu_op_mem {

template <typename T>
__global__ void scal_kernel(int n,T alpha,T* x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        x[idx] = alpha * x[idx];
    }
}

template <typename T>
Scal<T>::Scal(T alpha) : alpha_(alpha) {}

template <typename T>
void Scal<T::SetAlpha(T alpha) {
    alpha_ = alpha;
}

template <typename T>
StatusCode Scal<T>::Forward(Tensor<T>& x) {
    int n = static_cast<int>(n.numel());
    T* d_x = x.data();

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scal_kernel<T<<<blocks,threads>>>(n,alpha_,d_x);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR)<<"scal kernel failed: "<<cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    VLOG(1)<<"Scal: x = "<<alpha_<<" * x,n = "<<n;
    return StatusCode::SUCCESS;
}

template class Scal<float>;
template class Scal<double>;

}