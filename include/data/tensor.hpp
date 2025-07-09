#ifndef CUDA_TENSOR_H_
#define CUDA_TENSOR_H_

#include <vector>
#include <cstddef>
#include <numeric>
#include <type_traits>
#include <glog/logging.h>
#include "base/memory_pool.hpp"
#include <memory>

namespace cu_op_mem{

template <typename T>
class Tensor{
    static_assert(std::is_trivial<T>::value,"Tensor only support trivial types");

    public:
        Tensor() = default;

        explicit Tensor(const std::vector<std::size_t>& shape) : shape_(shape),
            numel_(compute_numel(shape)){
            
            if(numel_ == 0){
                VLOG(1)<<"Create an empty tensor of type "<<typeid(T).name();
                return;
            }
            data_ = static_cast<T*>(
                CudaMemoryPool::Instance().Alloc(bytes())
            );
            if(!data_){
                LOG(ERROR)<<"Tensor<"<<typeid(T).name()<<"> allocate GPU memory failed, bytes="<<bytes();
                throw std::runtime_error("Tensor allocate GPU memory failed");
            }
            VLOG(1)<<"Tensor<"<<typeid(T).name()<<"> allocate "<<bytes()<<" bytes, numel="<<numel_;
        }

        // 支持外部显存指针构造（不拥有内存）
        Tensor(T* external_data, const std::vector<std::size_t>& shape) : data_(external_data), shape_(shape), numel_(compute_numel(shape)) {}

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
            if(data_ && own_data_){
                CudaMemoryPool::Instance().Free(data_,bytes());
                VLOG(1)<<"Tensor<"<<typeid(T).name()<<"> release "<<bytes()<<" bytes";
            }
        }

        T* data() const { return data_; }

        std::size_t numel() const { return numel_; }

        std::size_t bytes() const { return numel_ * sizeof(T); }

        const std::vector<std::size_t>& shape() const { return shape_; }

        void reshape(const std::vector<std::size_t>& new_shape){
            std::size_t new_numel = compute_numel(new_shape);
            if(new_numel != numel_){
                LOG(ERROR)<<"Tensor<"<<typeid(T).name()<<"> reshape failed: numel not match, old="<<numel_<<", new="<<new_numel;
                throw std::runtime_error("Reshape total number not match");
            }
            shape_ = new_shape;
            VLOG(2)<<"Tensor<"<<typeid(T).name()<<"> reshape to ["<<join_shape(shape_)<<"]";
        }

        // 判断是否为空
        bool is_empty() const { return data_ == nullptr || numel_ == 0; }

        // 重置形状并重新分配显存（如有必要）
        void resize(const std::vector<std::size_t>& new_shape) {
            std::size_t new_numel = compute_numel(new_shape);
            if (new_numel == numel_) {
                shape_ = new_shape;
                return;
            }
            if (data_) {
                CudaMemoryPool::Instance().Free(data_, bytes());
                VLOG(1)<<"Tensor<"<<typeid(T).name()<<"> resize: free old "<<bytes()<<" bytes";
            }
            shape_ = new_shape;
            numel_ = new_numel;
            if (numel_ > 0) {
                data_ = static_cast<T*>(CudaMemoryPool::Instance().Alloc(bytes()));
                if (!data_) {
                    LOG(ERROR)<<"Tensor<"<<typeid(T).name()<<"> resize allocate GPU memory failed, bytes="<<bytes();
                    throw std::runtime_error("Tensor resize allocate GPU memory failed");
                }
                VLOG(1)<<"Tensor<"<<typeid(T).name()<<"> resize: allocate "<<bytes()<<" bytes";
            } else {
                data_ = nullptr;
            }
        }

        // 填充为指定值
        void fill(const T& value) {
            if (is_empty()) return;
            std::vector<T> host(numel_, value);
            cudaMemcpy(data_, host.data(), bytes(), cudaMemcpyHostToDevice);
        }

        // 全零
        void zero() {
            if (is_empty()) return;
            cudaMemset(data_, 0, bytes());
        }

        // 全一（仅适用于T为float/int等）
        void ones() {
            if (is_empty()) return;
            std::vector<T> host(numel_, static_cast<T>(1));
            cudaMemcpy(data_, host.data(), bytes(), cudaMemcpyHostToDevice);
        }

        // 从host拷贝
        void copy_from_host(const T* host_data, cudaStream_t stream = 0) {
            if (is_empty()) return;
            if (stream) {
                cudaMemcpyAsync(data_, host_data, bytes(), cudaMemcpyHostToDevice, stream);
            } else {
                cudaMemcpy(data_, host_data, bytes(), cudaMemcpyHostToDevice);
            }
        }

        // 拷贝到host
        void copy_to_host(T* host_data, cudaStream_t stream = 0) const {
            if (is_empty()) return;
            if (stream) {
                cudaMemcpyAsync(host_data, data_, bytes(), cudaMemcpyDeviceToHost, stream);
            } else {
                cudaMemcpy(host_data, data_, bytes(), cudaMemcpyDeviceToHost);
            }
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
        bool own_data_ = true; // 是否拥有内存
        std::shared_ptr<void> mem_holder_ = nullptr; // 显存生命周期管理

    public:
        // 一维切片（不拷贝数据，返回视图，显存复用）
        std::shared_ptr<Tensor<T>> slice(std::size_t start, std::size_t end) const {
            if (start >= end || end > numel_) throw std::out_of_range("slice out of range");
            auto t = std::make_shared<Tensor<T>>();
            t->mem_holder_ = mem_holder_ ? mem_holder_ : std::shared_ptr<void>(data_, [](void*){});
            t->data_ = data_ + start;
            t->shape_ = {end - start};
            t->numel_ = end - start;
            t->own_data_ = false;
            return t;
        }

        // 多维视图（offset为每维起始，shape为新形状，不拷贝数据，显存复用）
        std::shared_ptr<Tensor<T>> view(const std::vector<std::size_t>& offset, const std::vector<std::size_t>& new_shape) const {
            if (offset.size() != shape_.size() || new_shape.size() != shape_.size())
                throw std::invalid_argument("view: offset/shape dim mismatch");
            std::size_t flat_offset = 0, stride = 1;
            for (int i = (int)shape_.size() - 1; i >= 0; --i) {
                if (offset[i] + new_shape[i] > shape_[i])
                    throw std::out_of_range("view: out of range");
                flat_offset += offset[i] * stride;
                stride *= shape_[i];
            }
            auto t = std::make_shared<Tensor<T>>();
            t->mem_holder_ = mem_holder_ ? mem_holder_ : std::shared_ptr<void>(data_, [](void*){});
            t->data_ = data_ + flat_offset;
            t->shape_ = new_shape;
            t->numel_ = compute_numel(new_shape);
            t->own_data_ = false;
            return t;
        }

        // 类型转换（拷贝数据并转换类型，返回shared_ptr）
        template<typename U>
        std::shared_ptr<Tensor<U>> astype() const {
            auto t = std::make_shared<Tensor<U>>(shape_);
            if (numel_ > 0) {
                std::vector<T> host_src(numel_);
                cudaMemcpy(host_src.data(), data_, bytes(), cudaMemcpyDeviceToHost);
                std::vector<U> host_dst(numel_);
                std::transform(host_src.begin(), host_src.end(), host_dst.begin(), [](const T& v){ return static_cast<U>(v); });
                cudaMemcpy(t->data(), host_dst.data(), numel_ * sizeof(U), cudaMemcpyHostToDevice);
            }
            return t;
        }

};

}

#endif