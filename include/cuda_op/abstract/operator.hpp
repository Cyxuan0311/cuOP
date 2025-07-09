#ifndef OPERATOR_H_
#define OPERATOR_H_

#include "data/tensor.hpp"
#include "util/status_code.hpp"

namespace cu_op_mem{

template <typename T>
class Operator{
    public:
        virtual ~Operator() = default;

        virtual StatusCode Forward(const Tensor<T>& input,Tensor<T>& output) = 0;
};

}


#endif