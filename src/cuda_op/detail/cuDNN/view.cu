#include "cuda_op/detail/cuDNN/view.hpp"
#include <glog/logging.h>

namespace cu_op_mem {

template <typename T>
StatusCode View<T>::Forward(const Tensor<T>& input, Tensor<T>& output, const std::vector<std::size_t>& offset, const std::vector<std::size_t>& new_shape) {
    // 对于简单的reshape操作，我们直接复制数据
    // 这里实现一个简单的view操作：将输入张量重塑为新的形状
    
    // 计算总元素数量
    size_t total_elements = 1;
    for (size_t dim : new_shape) {
        total_elements *= dim;
    }
    
    // 验证元素数量匹配
    size_t input_elements = 1;
    for (size_t dim : input.shape()) {
        input_elements *= dim;
    }
    
    if (total_elements != input_elements) {
        LOG(ERROR) << "View: element count mismatch, input=" << input_elements << ", new_shape=" << total_elements;
        return StatusCode::SHAPE_MISMATCH;
    }
    
    // 确保输出张量有正确的形状
    if (output.shape() != new_shape) {
        LOG(ERROR) << "View: output shape mismatch, expected=" << new_shape.size() << " dims, got=" << output.shape().size() << " dims";
        return StatusCode::SHAPE_MISMATCH;
    }
    
    // 直接复制数据（对于view操作，数据在内存中是连续的）
    cudaError_t err = cudaMemcpy(output.data(), input.data(), total_elements * sizeof(T), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        LOG(ERROR) << "View: cudaMemcpy failed: " << cudaGetErrorString(err);
        return StatusCode::UNKNOWN_ERROR;
    }
    
    return StatusCode::SUCCESS;
}

// 显式实例化
template class View<float>;
template class View<double>;

} // namespace cu_op_mem