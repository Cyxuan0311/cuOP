#include "cuda_op/detail/cuDNN/flatten.hpp"
#include <glog/logging.h>

namespace cu_op_mem {

template <typename T>
StatusCode Flatten<T>::Forward(const Tensor<T>& input,Tensor<T>& output,int batch_dim) {
    const auto& in_shape = input.shape();
    if(in_shape.empty()) {
        LOG(ERROR)<<"Flatten: input shape is empty";
        return StatusCode::SHAPE_MISMATCH;
    }
    std::vector<std::size_t> out_shape;
    if(batch_dim == -1){
        out_shape = {input.numel()};
    }else{
        std::size_t batch = 1;
        for(int i = 0;i <= batch_dim && i < (int)in_shape.size(); ++i) batch *= in_shape[i];
        std::size_t features = input.numel() / batch;
        out_shape = {batch,features};
    }

    if(output.data() == nullptr || output.numel() != input.numel()) {
        output = Tensor<T>(out_shape);
        cudaMemcpy(output.data(),input.data(),input.bytes(),cudaMemcpyDeviceToDevice);
    }else {
        output.reshape(out_shape);
    }
    return StatusCode::SUCCESS;
}

template class Flatten<float>;
template class Flatten<double>;

}