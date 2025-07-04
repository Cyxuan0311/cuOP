#ifndef OPERATOR_H_
#define OPERATOR_H_

#include "data/tensor.hpp"

namespace cu_op_mem{

template <typename T>
class Operator{
    public:
        virtual ~Operator() = default;

        virtual void Forward(const Tensor<T>& input,Tensor<T>& output) = 0;
};

}


#endif