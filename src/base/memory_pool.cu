#include "../include/base/memory_pool.hpp"
#include <vector>
#include <algorithm>
#include <thread>
#include <sstream>
#include <iomanip>

namespace cu_op_mem {

// 线程局部变量定义
thread_local std::unordered_map<std::size_t, std::deque<MemoryBlock>> CudaMemoryPool::thread_free_list_;
thread_local std::size_t CudaMemoryPool::thread_cache_size_ = 0;

// 预定义的块大小（优化常见分配）
const std::vector<std::size_t> kBlockSizes = {
    256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576
};

CudaMemoryPool::CudaMemoryPool() 
    : last_defrag_time_(std::chrono::steady_clock::now()) {
    // 初始化配置
    config_.alignment = 256;
    config_.thread_cache_max_blocks = 64;
    config_.global_cache_max_blocks = 1024;
    config_.batch_alloc_count = 16;
    config_.defrag_threshold = 0.3;
    config_.defrag_interval_ms = 5000;
    config_.enable_stats = true;
    config_.enable_defrag = true;
    
    // 预分配常用大小的内存块
    for (auto size : kBlockSizes) {
        Preallocate(size, 4); // 每个大小预分配4个块
    }
    
    LOG(INFO) << "CudaMemoryPool initialized with enhanced features";
}

CudaMemoryPool& CudaMemoryPool::Instance() {
    static CudaMemoryPool instance;
    return instance;
}

CudaMemoryPool::~CudaMemoryPool() {
    ReleaseAll();
    LOG(INFO) << "CudaMemoryPool destroyed, all memory released";
}

std::size_t CudaMemoryPool::AlignSize(std::size_t size) const {
    return (size + config_.alignment - 1) / config_.alignment * config_.alignment;
}

std::size_t CudaMemoryPool::GetBlockSize(std::size_t size) const {
    for (auto block_size : kBlockSizes) {
        if (size <= block_size) return block_size;
    }
    return AlignSize(size); // 超大块，按对齐大小分配
}

void CudaMemoryPool::SetConfig(const MemoryPoolConfig& config) {
    std::lock_guard<std::mutex> lk(global_mutex_);
    config_ = config;
    LOG(INFO) << "MemoryPool config updated";
}

MemoryPoolConfig CudaMemoryPool::GetConfig() const {
    return config_;
}

MemoryStats CudaMemoryPool::GetStats() const {
    return stats_;
}

void CudaMemoryPool::ResetStats() {
    stats_.Reset();
    LOG(INFO) << "MemoryPool stats reset";
}

void CudaMemoryPool::PrintStats() const {
    auto stats = GetStats();
    std::stringstream ss;
    ss << "\n=== MemoryPool Statistics ===\n";
    ss << "Total Allocated: " << stats.total_allocated.load() << " bytes\n";
    ss << "Total Freed: " << stats.total_freed.load() << " bytes\n";
    ss << "Current Usage: " << stats.current_usage.load() << " bytes\n";
    ss << "Peak Usage: " << stats.peak_usage.load() << " bytes\n";
    ss << "Fragmentation: " << stats.fragmentation_bytes.load() << " bytes\n";
    ss << "Allocation Count: " << stats.allocation_count.load() << "\n";
    ss << "Free Count: " << stats.free_count.load() << "\n";
    ss << "Cache Hit Rate: " << std::fixed << std::setprecision(2)
       << (stats.allocation_count.load() > 0 ? 
           (double)stats.cache_hits.load() / stats.allocation_count.load() * 100 : 0)
       << "%\n";
    ss << "============================\n";
    LOG(INFO) << ss.str();
}

void CudaMemoryPool::Preallocate(std::size_t block_size, std::size_t count) {
    std::lock_guard<std::mutex> lk(global_mutex_);
    auto& pool = preallocated_pools_[block_size];
    for (std::size_t i = 0; i < count; ++i) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, block_size);
        if (err == cudaSuccess && ptr) {
            pool.push_back(ptr);
        } else {
            LOG(ERROR) << "Preallocate failed for size " << block_size 
                      << ": " << cudaGetErrorString(err);
        }
    }
    LOG(INFO) << "Preallocated " << count << " blocks of size " << block_size;
}

void* CudaMemoryPool::Alloc(std::size_t size) {
    return Alloc(size, 0); // 默认使用同步流
}

void* CudaMemoryPool::Alloc(std::size_t size, cudaStream_t stream) {
    if (size == 0) return nullptr;
    
    std::size_t aligned_size = AlignSize(size);
    std::size_t block_size = GetBlockSize(aligned_size);
    
    void* ptr = nullptr;
    bool cache_hit = false;
    
    // 1. 尝试从线程缓存分配
    ptr = AllocFromThreadCache(block_size);
    if (ptr) {
        cache_hit = true;
        VLOG(2) << "[ThreadCache] Allocated " << block_size << " bytes, ptr = " << ptr;
    }
    
    // 2. 尝试从全局缓存分配
    if (!ptr) {
        ptr = AllocFromGlobalCache(block_size);
        if (ptr) {
            cache_hit = true;
            VLOG(2) << "[GlobalCache] Allocated " << block_size << " bytes, ptr = " << ptr;
        }
    }
    
    // 3. 尝试从预分配池分配
    if (!ptr) {
        std::lock_guard<std::mutex> lk(global_mutex_);
        auto it = preallocated_pools_.find(block_size);
        if (it != preallocated_pools_.end() && !it->second.empty()) {
            ptr = it->second.front();
            it->second.pop_front();
            cache_hit = true;
            VLOG(2) << "[Preallocated] Allocated " << block_size << " bytes, ptr = " << ptr;
        }
    }
    
    // 4. 从设备直接分配
    if (!ptr) {
        ptr = AllocFromDevice(block_size);
        VLOG(2) << "[Device] Allocated " << block_size << " bytes, ptr = " << ptr;
    }
    
    if (ptr) {
        // 记录分配信息
        {
            std::lock_guard<std::mutex> lk(global_mutex_);
            allocated_blocks_[ptr] = MemoryBlock(ptr, size, block_size);
        }
        
        // 更新统计
        UpdateStats(aligned_size, 0, cache_hit);
        
        // 检查是否需要碎片整理
        if (config_.enable_defrag) {
            CheckAndDefragment();
        }
    }
    
    return ptr;
}

void* CudaMemoryPool::AllocFromThreadCache(std::size_t block_size) {
    auto& tlist = thread_free_list_[block_size];
    if (!tlist.empty()) {
        MemoryBlock block = tlist.front();
        tlist.pop_front();
        thread_cache_size_ -= block.block_size;
        return block.ptr;
    }
    return nullptr;
}

void* CudaMemoryPool::AllocFromGlobalCache(std::size_t block_size) {
    std::lock_guard<std::mutex> lk(global_mutex_);
    auto& glist = global_free_list_[block_size];
    if (!glist.empty()) {
        MemoryBlock block = glist.front();
        glist.pop_front();
        return block.ptr;
    }
    return nullptr;
}

void* CudaMemoryPool::AllocFromDevice(std::size_t block_size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, block_size);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMalloc failed for size " << block_size 
                  << ": " << cudaGetErrorString(err);
        return nullptr;
    }
    return ptr;
}

void CudaMemoryPool::Free(void* ptr, std::size_t size) {
    Free(ptr, size, 0); // 默认使用同步流
}

void CudaMemoryPool::Free(void* ptr, std::size_t size, cudaStream_t stream) {
    if (ptr == nullptr || size == 0) {
        LOG(WARNING) << "Free called with null pointer or zero size";
        return;
    }
    
    std::size_t aligned_size = AlignSize(size);
    std::size_t block_size = GetBlockSize(aligned_size);
    
    // 从已分配列表中移除
    MemoryBlock block(ptr, size, block_size);
    {
        std::lock_guard<std::mutex> lk(global_mutex_);
        auto it = allocated_blocks_.find(ptr);
        if (it != allocated_blocks_.end()) {
            block = it->second;
            allocated_blocks_.erase(it);
        }
    }
    
    // 返回到线程缓存
    ReturnToThreadCache(ptr, block_size);
    
    // 更新统计
    UpdateStats(0, aligned_size, false);
    
    VLOG(2) << "[Free] Freed " << block_size << " bytes, ptr = " << ptr;
}

void CudaMemoryPool::ReturnToThreadCache(void* ptr, std::size_t block_size) {
    auto& tlist = thread_free_list_[block_size];
    tlist.emplace_back(ptr, block_size, block_size);
    thread_cache_size_ += block_size;
    
    // 如果线程缓存过大，批量返回到全局缓存
    if (tlist.size() > config_.thread_cache_max_blocks) {
        std::lock_guard<std::mutex> lk(global_mutex_);
        auto& glist = global_free_list_[block_size];
        std::size_t return_count = std::min(tlist.size() - config_.thread_cache_max_blocks / 2, 
                                           config_.batch_alloc_count);
        
        for (std::size_t i = 0; i < return_count; ++i) {
            glist.push_back(tlist.front());
            tlist.pop_front();
            thread_cache_size_ -= block_size;
        }
        
        VLOG(2) << "[ThreadCache] Returned " << return_count << " blocks to global cache";
    }
}

void CudaMemoryPool::ReturnToGlobalCache(void* ptr, std::size_t block_size) {
    std::lock_guard<std::mutex> lk(global_mutex_);
    auto& glist = global_free_list_[block_size];
    glist.emplace_back(ptr, block_size, block_size);
    
    // 如果全局缓存过大，释放一些块
    if (glist.size() > config_.global_cache_max_blocks) {
        std::size_t evict_count = glist.size() - config_.global_cache_max_blocks / 2;
        for (std::size_t i = 0; i < evict_count; ++i) {
            void* evict_ptr = glist.front().ptr;
            glist.pop_front();
            cudaFree(evict_ptr);
        }
        VLOG(2) << "[GlobalCache] Evicted " << evict_count << " blocks";
    }
}

void CudaMemoryPool::BatchAllocate(std::size_t block_size, std::size_t batch_count) {
    std::lock_guard<std::mutex> lk(global_mutex_);
    auto& glist = global_free_list_[block_size];
    
    for (std::size_t i = 0; i < batch_count; ++i) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, block_size);
        if (err == cudaSuccess && ptr) {
            glist.emplace_back(ptr, block_size, block_size);
        } else {
            LOG(ERROR) << "BatchAllocate failed: " << cudaGetErrorString(err);
        }
    }
    
    VLOG(2) << "[BatchAllocate] Allocated " << batch_count << " blocks of size " << block_size;
}

void CudaMemoryPool::CheckAndDefragment() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_defrag_time_);
    
    if (elapsed.count() < config_.defrag_interval_ms) {
        return;
    }
    
    // 检查碎片率
    std::size_t total_fragmented = stats_.fragmentation_bytes.load();
    std::size_t total_allocated = stats_.total_allocated.load();
    
    if (total_allocated > 0) {
        double fragmentation_rate = static_cast<double>(total_fragmented) / total_allocated;
        if (fragmentation_rate > config_.defrag_threshold) {
            Defragment();
        }
    }
    
    last_defrag_time_ = now;
}

void CudaMemoryPool::Defragment() {
    if (defrag_in_progress_.exchange(true)) {
        return; // 已经在进行碎片整理
    }
    
    LOG(INFO) << "Starting memory defragmentation...";
    
    // 合并相邻块
    MergeAdjacentBlocks();
    
    // 压缩碎片块
    CompactFragmentedBlocks();
    
    // 清理过期块
    EvictOldBlocks();
    
    defrag_in_progress_ = false;
    LOG(INFO) << "Memory defragmentation completed";
}

void CudaMemoryPool::MergeAdjacentBlocks() {
    std::lock_guard<std::mutex> lk(global_mutex_);
    
    for (auto& kv : global_free_list_) {
        auto& blocks = kv.second;
        if (blocks.size() < 2) continue;
        
        // 按地址排序
        std::sort(blocks.begin(), blocks.end(), 
                 [](const MemoryBlock& a, const MemoryBlock& b) {
                     return a.ptr < b.ptr;
                 });
        
        // 尝试合并相邻块
        for (auto it = blocks.begin(); it != blocks.end() - 1;) {
            auto next_it = it + 1;
            char* current_end = static_cast<char*>(it->ptr) + it->block_size;
            
            if (current_end == static_cast<char*>(next_it->ptr)) {
                // 可以合并
                it->block_size += next_it->block_size;
                blocks.erase(next_it);
            } else {
                ++it;
            }
        }
    }
}

void CudaMemoryPool::CompactFragmentedBlocks() {
    std::lock_guard<std::mutex> lk(global_mutex_);
    
    // 统计碎片
    std::size_t fragmented_bytes = 0;
    for (const auto& kv : global_free_list_) {
        for (const auto& block : kv.second) {
            if (block.is_fragmented) {
                fragmented_bytes += block.block_size;
            }
        }
    }
    
    stats_.fragmentation_bytes.store(fragmented_bytes);
}

void CudaMemoryPool::EvictOldBlocks() {
    auto now = std::chrono::steady_clock::now();
    auto max_age = std::chrono::seconds(30); // 30秒未使用的块将被释放
    
    std::lock_guard<std::mutex> lk(global_mutex_);
    
    for (auto& kv : global_free_list_) {
        auto& blocks = kv.second;
        auto it = blocks.begin();
        while (it != blocks.end()) {
            auto age = now - it->last_used;
            if (age > max_age && blocks.size() > 4) { // 保留至少4个块
                cudaFree(it->ptr);
                it = blocks.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void CudaMemoryPool::TrimCache() {
    TrimThreadCache();
    TrimGlobalCache();
}

void CudaMemoryPool::TrimThreadCache() {
    // 线程缓存由ReturnToThreadCache自动管理
    VLOG(2) << "[TrimCache] Thread cache trimmed";
}

void CudaMemoryPool::TrimGlobalCache() {
    std::lock_guard<std::mutex> lk(global_mutex_);
    
    for (auto& kv : global_free_list_) {
        auto& blocks = kv.second;
        if (blocks.size() > config_.global_cache_max_blocks / 2) {
            std::size_t evict_count = blocks.size() - config_.global_cache_max_blocks / 4;
            for (std::size_t i = 0; i < evict_count; ++i) {
                cudaFree(blocks.front().ptr);
                blocks.pop_front();
            }
        }
    }
    
    VLOG(2) << "[TrimCache] Global cache trimmed";
}

void CudaMemoryPool::UpdateStats(std::size_t allocated, std::size_t freed, bool cache_hit) {
    if (!config_.enable_stats) return;
    
    if (allocated > 0) {
        stats_.total_allocated.fetch_add(allocated);
        stats_.current_usage.fetch_add(allocated);
        stats_.allocation_count.fetch_add(1);
        
        // 更新峰值使用量
        std::size_t current = stats_.current_usage.load();
        std::size_t peak = stats_.peak_usage.load();
        while (current > peak && !stats_.peak_usage.compare_exchange_weak(peak, current)) {
            // 自旋直到更新成功
        }
    }
    
    if (freed > 0) {
        stats_.total_freed.fetch_add(freed);
        stats_.current_usage.fetch_sub(freed);
        stats_.free_count.fetch_add(1);
    }
    
    if (cache_hit) {
        stats_.cache_hits.fetch_add(1);
    } else if (allocated > 0) {
        stats_.cache_misses.fetch_add(1);
    }
}

void CudaMemoryPool::ReleaseAll() {
    std::lock_guard<std::mutex> lk(global_mutex_);
    
    // 释放全局缓存
    for (auto& kv : global_free_list_) {
        for (auto& block : kv.second) {
            cudaError_t err = cudaFree(block.ptr);
            if (err != cudaSuccess) {
                LOG(ERROR) << "ReleaseAll failed: " << cudaGetErrorString(err);
            }
        }
        kv.second.clear();
    }
    global_free_list_.clear();
    
    // 释放预分配池
    for (auto& kv : preallocated_pools_) {
        for (void* ptr : kv.second) {
            cudaFree(ptr);
        }
        kv.second.clear();
    }
    preallocated_pools_.clear();
    
    // 释放流缓存
    for (auto& stream_kv : stream_caches_) {
        for (auto& size_kv : stream_kv.second) {
            for (void* ptr : size_kv.second) {
                cudaFree(ptr);
            }
            size_kv.second.clear();
        }
    }
    stream_caches_.clear();
    
    // 释放已分配块（不应该有）
    for (auto& kv : allocated_blocks_) {
        LOG(WARNING) << "Found allocated block during ReleaseAll: " << kv.first;
        cudaFree(kv.first);
    }
    allocated_blocks_.clear();
    
    // 重置统计
    stats_.Reset();
    
    LOG(INFO) << "All memory released from CudaMemoryPool";
}

} // namespace cu_op_mem


