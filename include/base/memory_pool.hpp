#ifndef BASE_MEMORY_POOL_HPP
#define BASE_MEMORY_POOL_HPP

#include <cuda_runtime.h>
#include <map>
#include <list>
#include <mutex>
#include <cstddef>
#include <glog/logging.h>
#include <deque>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <vector>
#include <memory>
#include <chrono>

namespace cu_op_mem {

// 内存块信息结构
struct MemoryBlock {
    void* ptr;
    std::size_t size;
    std::size_t block_size;
    std::chrono::steady_clock::time_point last_used;
    bool is_fragmented;
    
    MemoryBlock(void* p, std::size_t s, std::size_t bs) 
        : ptr(p), size(s), block_size(bs), last_used(std::chrono::steady_clock::now()), is_fragmented(false) {}
};

// 内存统计信息
struct MemoryStats {
    std::atomic<std::size_t> total_allocated{0};
    std::atomic<std::size_t> total_freed{0};
    std::atomic<std::size_t> peak_usage{0};
    std::atomic<std::size_t> current_usage{0};
    std::atomic<std::size_t> fragmentation_bytes{0};
    std::atomic<std::size_t> allocation_count{0};
    std::atomic<std::size_t> free_count{0};
    std::atomic<std::size_t> cache_hits{0};
    std::atomic<std::size_t> cache_misses{0};
    
    void Reset() {
        total_allocated = 0;
        total_freed = 0;
        peak_usage = 0;
        current_usage = 0;
        fragmentation_bytes = 0;
        allocation_count = 0;
        free_count = 0;
        cache_hits = 0;
        cache_misses = 0;
    }
};

// 内存池配置
struct MemoryPoolConfig {
    std::size_t alignment = 256;
    std::size_t thread_cache_max_blocks = 64;
    std::size_t global_cache_max_blocks = 1024;
    std::size_t batch_alloc_count = 16;
    std::size_t defrag_threshold = 0.3; // 30%碎片率触发整理
    std::size_t defrag_interval_ms = 5000; // 5秒检查一次
    bool enable_stats = true;
    bool enable_defrag = true;
};

class CudaMemoryPool {
public:
    static CudaMemoryPool& Instance();
    
    // 基础接口（保持向后兼容）
    void* Alloc(std::size_t size);
    void Free(void* ptr, std::size_t size);
    void ReleaseAll();
    
    // 增强接口
    void* Alloc(std::size_t size, cudaStream_t stream);
    void Free(void* ptr, std::size_t size, cudaStream_t stream);
    
    // 配置接口
    void SetConfig(const MemoryPoolConfig& config);
    MemoryPoolConfig GetConfig() const;
    
    // 统计接口
    MemoryStats GetStats() const;
    void ResetStats();
    void PrintStats() const;
    
    // 内存管理接口
    void Defragment();
    void TrimCache();
    void Preallocate(std::size_t block_size, std::size_t count);
    
    // 线程安全
    CudaMemoryPool(const CudaMemoryPool&) = delete;
    CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

private:
    CudaMemoryPool();
    ~CudaMemoryPool();
    
    // 内部辅助函数
    std::size_t AlignSize(std::size_t size) const;
    std::size_t GetBlockSize(std::size_t size) const;
    void BatchAllocate(std::size_t block_size, std::size_t batch_count);
    
    // 多级缓存管理
    void* AllocFromThreadCache(std::size_t block_size);
    void* AllocFromGlobalCache(std::size_t block_size);
    void* AllocFromDevice(std::size_t block_size);
    void ReturnToThreadCache(void* ptr, std::size_t block_size);
    void ReturnToGlobalCache(void* ptr, std::size_t block_size);
    
    // 碎片整理
    void CheckAndDefragment();
    void MergeAdjacentBlocks();
    void CompactFragmentedBlocks();
    
    // 缓存管理
    void TrimThreadCache();
    void TrimGlobalCache();
    void EvictOldBlocks();
    
    // 统计更新
    void UpdateStats(std::size_t allocated, std::size_t freed, bool cache_hit);
    
    // 成员变量
    MemoryPoolConfig config_;
    MemoryStats stats_;
    
    // 全局缓存（进程级别）
    mutable std::mutex global_mutex_;
    std::unordered_map<std::size_t, std::deque<MemoryBlock>> global_free_list_;
    std::unordered_map<void*, MemoryBlock> allocated_blocks_;
    
    // 线程局部缓存
    thread_local static std::unordered_map<std::size_t, std::deque<MemoryBlock>> thread_free_list_;
    thread_local static std::size_t thread_cache_size_;
    
    // 碎片整理相关
    std::chrono::steady_clock::time_point last_defrag_time_;
    std::atomic<bool> defrag_in_progress_{false};
    
    // 预分配缓存
    std::vector<std::size_t> preallocated_sizes_;
    std::unordered_map<std::size_t, std::deque<void*>> preallocated_pools_;
    
    // 流相关缓存
    std::unordered_map<cudaStream_t, std::unordered_map<std::size_t, std::deque<void*>>> stream_caches_;
    mutable std::mutex stream_mutex_;
};

} // namespace cu_op_mem

#endif