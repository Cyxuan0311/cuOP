#include "../include/base/memory_pool.hpp"
#include <vector>
#include <algorithm>
#include <thread>

namespace cu_op_mem{

std::mutex CudaMemoryPool::global_mutex_;
std::unordered_map<std::size_t, std::deque<void*>> CudaMemoryPool::global_free_list_;
thread_local std::unordered_map<std::size_t, std::deque<void*>> CudaMemoryPool::thread_free_list_;

constexpr std::size_t ALIGN_BYTES = 256;
const std::vector<std::size_t> kBlockSizes = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
constexpr std::size_t kBatchAllocCount = 8; // 每次批量分配的块数
constexpr std::size_t kThreadCacheMax = 32; // 线程缓存最大块数

// 线程局部缓存
thread_local std::unordered_map<std::size_t, std::deque<void*>> thread_free_list_;

CudaMemoryPool& CudaMemoryPool::Instance(){
    static CudaMemoryPool instance;
    return instance;
}

CudaMemoryPool::~CudaMemoryPool(){
    ReleaseAll();
    LOG(INFO)<<"CudaMemoryPool has destory already,no memory exist!!!\n";
}

std::size_t CudaMemoryPool::AlignSize(std::size_t size) const {
    return (size + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
}

std::size_t CudaMemoryPool::GetBlockSize(std::size_t size) const {
    for (auto s : kBlockSizes) {
        if (size <= s) return s;
    }
    return size; // 超大块，直接分配
}

// 批量分配到全局池
void CudaMemoryPool::BatchAllocate(std::size_t block_size, std::size_t batch_count) {
    std::lock_guard<std::mutex> lk(global_mutex_);
    for (std::size_t i = 0; i < batch_count; ++i) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, block_size);
        if (err == cudaSuccess && ptr) {
            global_free_list_[block_size].push_back(ptr);
        } else {
            LOG(ERROR)<<"BatchAllocate failed, cudaMalloc error: "<<cudaGetErrorString(err);
        }
    }
}

void* CudaMemoryPool::Alloc(std::size_t size){
    std::size_t asize = AlignSize(size);
    std::size_t block_size = GetBlockSize(asize);

    // 1. 线程局部缓存
    auto& tlist = thread_free_list_[block_size];
    if (!tlist.empty()) {
        void* ptr = tlist.front();
        tlist.pop_front();
        LOG(INFO)<<"[ThreadCache] Alloc block: "<<block_size<<" bytes, ptr = "<<ptr;
        return ptr;
    }

    // 2. 全局池批量获取
    {
        std::lock_guard<std::mutex> lk(global_mutex_);
        auto& glist = global_free_list_[block_size];
        if (glist.size() >= kBatchAllocCount) {
            // 批量分配到线程缓存
            for (std::size_t i = 0; i < kBatchAllocCount-1; ++i) {
                tlist.push_back(glist.front());
                glist.pop_front();
            }
            void* ptr = glist.front();
            glist.pop_front();
            LOG(INFO)<<"[GlobalPool] Alloc block: "<<block_size<<" bytes, ptr = "<<ptr;
            return ptr;
        }
    }

    // 3. 还不够，直接批量 cudaMalloc
    for (std::size_t i = 0; i < kBatchAllocCount-1; ++i) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, block_size);
        if (err == cudaSuccess && ptr) {
            tlist.push_back(ptr);
        } else {
            LOG(ERROR)<<"Alloc failed, cudaMalloc error: "<<cudaGetErrorString(err);
        }
    }
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, block_size);
    if (err != cudaSuccess) {
        LOG(ERROR)<<"Alloc failed, cudaMalloc error: "<<cudaGetErrorString(err);
        return nullptr;
    }
    LOG(INFO)<<"[cudaMalloc] Alloc new block: "<<block_size<<" bytes, ptr = "<<ptr;
    return ptr;
}

void CudaMemoryPool::Free(void* ptr, std::size_t size){
    if(ptr == nullptr || size == 0){
        LOG(WARNING)<<"CudaMemoryPool::Free called with null pointer or zero size";
        return;
    }
    std::size_t asize = AlignSize(size);
    std::size_t block_size = GetBlockSize(asize);
    auto& tlist = thread_free_list_[block_size];
    tlist.push_back(ptr);
    LOG(INFO)<<"[ThreadCache] Free block: "<<block_size<<" bytes, ptr = "<<ptr;
    // 线程缓存过多，批量归还全局池
    if (tlist.size() > kThreadCacheMax) {
        std::lock_guard<std::mutex> lk(global_mutex_);
        auto& glist = global_free_list_[block_size];
        for (std::size_t i = 0; i < kBatchAllocCount && !tlist.empty(); ++i) {
            glist.push_back(tlist.front());
            tlist.pop_front();
        }
        LOG(INFO)<<"[ThreadCache] Batch return to GlobalPool, block: "<<block_size;
    }
}

void CudaMemoryPool::ReleaseAll(){
    std::lock_guard<std::mutex> lk(global_mutex_);
    for(auto& kv : global_free_list_){
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
    global_free_list_.clear();
    LOG(INFO)<<"All free bolck has release from device, Pool is clear now!!";
}

} // namespace cu_op_mem


