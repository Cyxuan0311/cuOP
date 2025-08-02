#include "cuda_op/detail/cuDNN/view.hpp"
#include <glog/logging.h>

namespace cu_op_mem {

template <typename T>
StatusCode View<T>::Forward(const Tensor<T>& input, Tensor<T>& output, const std::vector<std::size_t>& offset, const std::vector<std::size_t>& new_shape) {
    if (offset.size() != input.shape().size() || new_shape.size() != input.shape().size()) {
        LOG(ERROR) << "View: offset/new_shape dim mismatch, input dim=" << input.shape().size();
        return StatusCode::SHAPE_MISMATCH;
    }
    try {
        // Tensor::view 返回 shared_ptr<Tensor<T>>
        auto view_ptr = input.view(offset, new_shape);
        // output 赋值为 view 结果（数据指针和 shape 变了，显存复用）
        output = *view_ptr;
    } catch (const std::exception& e) {
        LOG(ERROR) << "View: exception: " << e.what();
        return StatusCode::UNKNOWN_ERROR;
    }
    return StatusCode::SUCCESS;
}

// 显式实例化
template class View<float>;
template class View<double>;

} // namespace cu_op_mem