#ifndef CUDA_TENSOR_H_
#define CUDA_TENSOR_H_

#include <vector>
#include <cstddef>
#include <numeric>
#include <type_traits>
#include <glog/logging.h>
#include "base/memory_pool.hpp"

namespace cu_op_mem{

template <typename T>
class Tensor{
    static_assert(std::is_trivial<T>::value,"Tensor only support trivial types");

    public:
        Tensor() = default;

        explicit Tensor(const std::vector<std::size_t>& shape) : shape_(shape),
            numel_(compute_numel(shape)){
            
            if(numel_ == 0){
                LOG(WARNING)<<"Create an empty tensor "<<typeid(T).name()<<">";
                return;
            }
            data_ = static_cast<T*>(
                CudaMemoryPool::Instance().Alloc(bytes())
            );
            if(!data_){
                throw std::runtime_error("Tensor allocate GPU memory failed");
            }
            LOG(INFO)<<"Tensor< "<<typeid(T).name()<<"> allocate "<<bytes()<<"bytes , element number "<<numel_;
        }

        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;

        Tensor(Tensor&& other) noexcept : data_(other.data_),
            shape_(std::move(other.shape_)),numel_(other.numel_){
            other.data_ = nullptr;
            other.numel_ = 0;
        }

        Tensor& operator=(Tensor&& other) noexcept{
            if(this != &other){
                if(data_){
                    CudaMemoryPool::Instance().Free(data_,bytes());
                }

                data_ = other.data_;
                shape_ = std::move(other.shape_);
                numel_ = other.numel_;

                other.data_ = nullptr;
                other.numel_ = 0;
            }
            return *this;
        }

        ~Tensor(){
            if(data_){
                CudaMemoryPool::Instance().Free(data_,bytes());
                LOG(INFO)<<"Tensor<"<<typeid(T).name()<<"> release"
                    <<bytes()<<" bytes";
            }
        }

        T* data() const { return data_; }

        std::size_t numel() const { return numel_; }

        std::size_t bytes() const { return numel_ * sizeof(T); }

        const std::vector<std::size_t>& shape() const { return shape_; }

        void reshape(const std::vector<std::size_t>& new_shape){
            std::size_t new_numel = compute_numel(new_shape);
            if(new_numel != numel_){
                throw std::runtime_error("Reshape total number not match");
            }
            shape_ = new_shape;
            LOG(INFO)<<"Tensor <"<<typeid(T).name()<<"> reshape util "<<" [ "<<
                join_shape(shape_)<<"]";
        }

    private:
        
        static std::size_t compute_numel(const std::vector<std::size_t>& shape){
            if(shape.empty()) return 0;
            return std::accumulate(
                shape.begin(),
                shape.end(),
                (std::size_t)1,
                std::multiplies<std::size_t>()
            );
        }

        static std::string join_shape(const std::vector<std::size_t>& shape){
            std::string s;
            for(size_t i = 0;i<shape.size();++i){
                s += std::to_string(shape[i]);
                if(i < shape.size() - 1){
                    s += "x";
                }
            }
            return s;
        }

        T* data_ = nullptr;
        std::vector<std::size_t> shape_;
        std::size_t numel_ = 0;

};

}

#endif