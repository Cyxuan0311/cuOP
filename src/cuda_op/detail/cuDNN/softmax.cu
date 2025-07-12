#include "cuda_op/detail/cuDNN/softmax.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>

namespace cu_op_mem {

template <typename T>
__device__ T warpReduceMax(T val){
    for(int offset = (warpSize / 2); offset > 0; offset /= 2)
        val = max(val,__shfl_down_sync(0xffffffff,val,offset));
    return val;
}

template <typename T>
__device__ T warpReduceSum(T val){
    for(int offset = (warpSize / 2); offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff,val,offset);
    return val;
}

template <typename T>
__device__ T blockReduceMax(T val){
    static __shared__ T shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceMax(val);

    if(lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -INFINITY;
    if(wid == 0) val = warpReduceMax(val);

    return val;
}

template <typename T>
__device__ T blockReduceSum(T val){ 
    static __shared__ T shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if(lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if(wid == 0) val = warpReduceSum(val);

    return val;
}

template <typename T>
__global__ void softmax_kernel(const T* in,T* out,std::size_t N){
    T thread_max = -INFINITY;
    for(std::size_t idx = threadIdx.x; idx < N; idx += blockDim.x) {
        thread_max = max(thread_max,in[idx]);
    }
    T max_val = blockReduceMax(thread_max);

    T thread_sum = 0;
    for(std::size_t idx = threadIdx.x; idx < N; idx += blockDim.x) {
        out[idx] = exp(in[idx] - max_val);
        thread_sum += out[idx];
    }
    T sum_val = blockReduceSum(thread_sum);

    //归一化
    for(std::size_t idx = threadIdx.x; idx < N; idx += blockDim.x) {
        out[idx] /= sum_val;
    }
}

template <typename T>
StatusCode Softmax<T>::Forward(const Tensor<T>& input,Tensor<T>& output) {
    std::size_t N = input.numel();

    if(output.data() == nullptr) {
        output = Tensor<T>(input.shape());
    }else if(output.numel() != N) {
        LOG(ERROR) << "Softmax's output and input tensor size not match";
        return StatusCode::SHAPE_MISMATCH;
    }

    const T* d_in = input.data();
    T* d_out = output.data();

    const std::size_t threads = 256;
    const std::size_t blocks = 1;
    LOG(INFO)<<"Softmax launch: blocks = "<<blocks<<" , threads = "<<threads<<" , N = "<<N;

    softmax_kernel<T><<<blocks, threads>>>(d_in, d_out, N);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        LOG(ERROR)<<"Softmax kernel failed: "<<cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }

    return StatusCode::SUCCESS;
}

}