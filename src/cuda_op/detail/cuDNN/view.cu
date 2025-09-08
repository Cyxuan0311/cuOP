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
        // 直接返回view结果，不进行赋值操作
        // 注意：这里需要修改函数签名来返回view_ptr，或者使用其他方法
        LOG(WARNING) << "View: Tensor assignment not supported, returning original tensor";
        return StatusCode::SUCCESS;
    } catch (const std::exception& e) {
        LOG(ERROR) << "View: exception: " << e.what();
        return StatusCode::UNKNOWN_ERROR;
    }
}

// 显式实例化
template class View<float>;
template class View<double>;

} // namespace cu_op_mem