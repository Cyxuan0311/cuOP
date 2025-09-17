#ifndef CUDA_OP_VIEW_OP_H_
#define CUDA_OP_VIEW_OP_H_

#include "cuda_op/abstract/operator.hpp"

namespace cu_op_mem {

template <typename T>
class View : public Operator<T> {
    private:
        std::vector<std::size_t> offset_;
        std::vector<std::size_t> new_shape_;
    
    public:
        View() = default;
        View(const std::vector<std::size_t>& offset, const std::vector<std::size_t>& new_shape) 
            : offset_(offset), new_shape_(new_shape) {}
        
        StatusCode Forward(const Tensor<T>& input,Tensor<T>& output,const std::vector<std::size_t>& offset,const std::vector<std::size_t>& new_shape);

        StatusCode Forward(const Tensor<T>& input,Tensor<T>& output) override {
            if (offset_.empty() || new_shape_.empty()) {
                // 如果没有设置参数，则使用输出张量的形状进行reshape
                std::vector<std::size_t> zero_offset(input.shape().size(), 0);
                return Forward(input, output, zero_offset, output.shape());
            }
            return Forward(input, output, offset_, new_shape_);
        }
        
        void SetOffset(const std::vector<std::size_t>& offset) { offset_ = offset; }
        void SetShape(const std::vector<std::size_t>& new_shape) { new_shape_ = new_shape; }
};

}

#endif