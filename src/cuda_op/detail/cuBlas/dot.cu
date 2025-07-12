#include "cuda_op/detail/cuBlas/dot.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace cu_op_mem {

    template <typename T>
    __global__ void dot_kernel(const T* x,const T* y,T* partial,int n){
        extern __shared__ T sdata[];
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        T val = 0;
        if (idx < n) val = x[idx] * y[idx];
        sdata[tid] = val;
        __syncthreads();

        //归约
        for(int s = blockDim.x / 2;s > 0;s>>=1){
            if(tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if(tid == 0) partial[blockIdx.x] = sdata[0];
    }

    template <typename T>
    Dot<T>::Dot() {}

    template <typename T>
    Dot<T>::~Dot() {}

    template <typename T>
    StatusCode Dot<T>::Forward(const Tensor<T>& x,const Tensor<T>& y,T& result) {
        if(x.numel() != y.numel()){
            LOG(ERROR)<<"Dot: x and y must have same number of elements";
            return StatusCode::SHAPE_MISMATCH;
        }
        int n = static_cast<int>(x.numel());
        const T* d_x = x.data();
        const T* d_y = y.data();

        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        T* d_partial = nullptr;
        cudaMalloc(&d_partial,blocks * sizeof(T));

        dot_kernel<T><<<blocks,threads,threads * sizeof(T)>>>(d_x,d_y,d_partial,n);

        std::vector<T> h_partial(blocks);
        cudaMemcpy(h_partial.data(),d_partial,blocks * sizeof(T),cudaMemcpyDeviceToHost);
        cudaFree(d_partial);

        result = 0;
        for (int i = 0;i < blocks; ++i) result += h_partial[i];

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR)<<" dot kernel failed: "<<cudaGetErrorString(err);
            return StatusCode::CUDA_ERROR;
        }
        VLOG(1)<<"Dot: result = x^T y,n = "<<n<<", result = "<<result;
        return StatusCode::SUCCESS;
    }

    template class Dot<float>;
    template class Dot<double>;

}