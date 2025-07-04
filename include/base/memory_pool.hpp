#ifndef BASE_MEMORY_POOL_HPP
#define BASE_MEMORY_POOL_HPP

#include <cuda_runtime.h>
#include <map>
#include <list>
#include <mutex>
#include <cstddef>
#include <glog/logging.h>

namespace cu_op_mem{

class CudaMemoryPool{
    public:
        static CudaMemoryPool& Instance();

        void* Alloc(std::size_t size);

        void Free(void* ptr,std::size_t size);

        //void Free(void* ptr,std::size_t size);

        void ReleaseAll();

        CudaMemoryPool(const CudaMemoryPool&) = delete;

        CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

    private:
        CudaMemoryPool() = default;
        ~CudaMemoryPool();

        std::size_t AlignSize(std::size_t size);

        std::map<std::size_t,std::list<void*>> free_list_;
        
        std::mutex mutex_;
};

}

#endif