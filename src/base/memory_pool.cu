#include "../include/base/memory_pool.hpp"

namespace cu_op_mem{

constexpr std::size_t ALIGN_BYTES = 256;

CudaMemoryPool& CudaMemoryPool::Instance(){
    static CudaMemoryPool instance;
    return instance;
}

CudaMemoryPool::~CudaMemoryPool(){
    ReleaseAll();
    LOG(INFO)<<"CudaMemoryPool has destory already,no memory exist!!!\n";
}

std::size_t CudaMemoryPool::AlignSize(std::size_t size){
    return (size + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
}

void* CudaMemoryPool::Alloc(std::size_t size){
    std::size_t asize = AlignSize(size);
    std::lock_guard<std::mutex> lk(mutex_);

    auto it = free_list_.find(asize);
    if(it != free_list_.end() && !it->second.empty()){
        void* ptr = it->second.front();;
        it->second.pop_front();
        LOG(INFO)<<"The block of memory: "<<asize<<" bytes , ptr = "<<ptr;
        return ptr;
    }

    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr,asize);
    if(err != cudaSuccess){
        LOG(ERROR)<<"CudaMemoryPool::Alloc failed, cudaMalloc error: "<<cudaGetErrorString(err);
        return nullptr;
    }
    LOG(INFO)<<"Alloc new block of memory: "<<asize<<" bytes , ptr = "<<ptr;
    return ptr;
}

void CudaMemoryPool::Free(void* ptr, std::size_t size){
    if(ptr == nullptr || size == 0){
        LOG(WARNING)<<"CudaMemoryPool::Free called with null pointer or zero size";
        return;
    }

    std::size_t asize = AlignSize(size);
    std::lock_guard<std::mutex> lk(mutex_);

    free_list_[asize].push_back(ptr);
    LOG(INFO)<<"Free memory block: "<<asize<<" bytes , ptr = "<<ptr;

}

void CudaMemoryPool::ReleaseAll(){
    std::lock_guard<std::mutex> lk(mutex_);
    for(auto& kv : free_list_){
        std::size_t blockSize = kv.first;
        for(void* ptr : kv.second){
            cudaError_t err = cudaFree(ptr);
            if(err != cudaSuccess){
                LOG(ERROR)<<"CudaMemoryPool::ReleaseAll failed, cudaFree error: "<<cudaGetErrorString(err);
            }else{
                LOG(ERROR)<<"Release the memory of device "<<blockSize<<" bytes, ptr = "<<ptr;
            }
        }
        kv.second.clear();
    }
    free_list_.clear();
    LOG(INFO)<<"All free bolck has release from device, Pool is clear now!!";
}

}


