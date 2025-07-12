#ifndef BASE_MEMORY_POOL_HPP
#define BASE_MEMORY_POOL_HPP

#include <cuda_runtime.h>
#include <map>
#include <list>
#include <mutex>
#include <cstddef>
#include <glog/logging.h>
#include <deque>
#include <unordered_map> // 补充
#include <thread>

namespace cu_op_mem {

class CudaMemoryPool {
public:
    static CudaMemoryPool& Instance();

    void* Alloc(std::size_t size);
    void Free(void* ptr, std::size_t size);
    void ReleaseAll();

    CudaMemoryPool(const CudaMemoryPool&) = delete;
    CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

    // 新增成员函数声明
    std::size_t AlignSize(std::size_t size) const;
    std::size_t GetBlockSize(std::size_t size) const;
    void BatchAllocate(std::size_t block_size, std::size_t batch_count);

private:
    CudaMemoryPool() = default;
    ~CudaMemoryPool();

    // 原有成员
    std::map<std::size_t, std::list<void*>> free_list_;
    std::mutex mutex_;

    // 新增静态成员变量声明
    static std::mutex global_mutex_;
    static std::unordered_map<std::size_t, std::deque<void*>> global_free_list_;
    thread_local static std::unordered_map<std::size_t, std::deque<void*>> thread_free_list_;
};

} // namespace cu_op_mem

#endif