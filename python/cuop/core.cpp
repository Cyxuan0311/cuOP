#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "data/tensor.hpp"
#include "cuda_op/detail/cuBlas/gemm.hpp"
#include "cuda_op/detail/cuBlas/gemv.hpp"
#include "cuda_op/detail/cuDNN/relu.hpp"
#include "cuda_op/detail/cuDNN/softmax.hpp"
#include "cuda_op/detail/cuDNN/matmul.hpp"
#include "jit/jit_wrapper.hpp"
#include "jit/jit_config.hpp"
#include "jit/jit_persistent_cache.hpp"
#include "util/status_code.hpp"

namespace py = pybind11;
using namespace cu_op_mem;

// 辅助函数：将numpy数组转换为Tensor
template<typename T>
Tensor<T> numpy_to_tensor(py::array_t<T, py::array::c_style> arr) {
    auto info = arr.request();
    
    if (info.ndim == 0) {
        throw std::runtime_error("Cannot create tensor from 0-dimensional array");
    }
    
    std::vector<size_t> shape;
    for (size_t i = 0; i < info.ndim; ++i) {
        shape.push_back(static_cast<size_t>(info.shape[i]));
    }
    
    Tensor<T> tensor(shape);
    
    // 复制数据到GPU
    if (info.size > 0) {
        cudaMemcpy(tensor.data(), info.ptr, info.size * sizeof(T), cudaMemcpyHostToDevice);
    }
    
    return tensor;
}

// 辅助函数：将Tensor转换为numpy数组
template<typename T>
py::array_t<T> tensor_to_numpy(const Tensor<T>& tensor) {
    auto shape = tensor.shape();
    std::vector<py::ssize_t> py_shape;
    for (auto dim : shape) {
        py_shape.push_back(static_cast<py::ssize_t>(dim));
    }
    
    auto result = py::array_t<T>(py_shape);
    auto info = result.request();
    
    // 复制数据到CPU
    if (info.size > 0) {
        cudaMemcpy(info.ptr, tensor.data(), info.size * sizeof(T), cudaMemcpyDeviceToHost);
    }
    
    return result;
}

// 辅助函数：检查CUDA错误
void check_cuda_error() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
    }
}

// Tensor类绑定
template<typename T>
void bind_tensor(py::module& m, const std::string& type_name) {
    py::class_<Tensor<T>>(m, ("Tensor" + type_name).c_str())
        .def(py::init<>())
        .def(py::init<const std::vector<size_t>&>())
        .def(py::init<const std::vector<size_t>&, T>())
        .def("shape", &Tensor<T>::shape)
        .def("size", &Tensor<T>::size)
        .def("dtype", [](const Tensor<T>& self) { return py::dtype::of<T>(); })
        .def("fill", &Tensor<T>::Fill)
        .def("zero", &Tensor<T>::Zero)
        .def("copy_from", [](Tensor<T>& self, py::array_t<T, py::array::c_style> arr) {
            auto tensor = numpy_to_tensor<T>(arr);
            self = tensor;
        })
        .def("to_numpy", &tensor_to_numpy<T>)
        .def("__repr__", [](const Tensor<T>& self) {
            std::stringstream ss;
            ss << "Tensor<" << typeid(T).name() << ">(";
            auto shape = self.shape();
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << shape[i];
            }
            ss << ")";
            return ss.str();
        });
}

// 基础算子绑定
template<typename T>
void bind_basic_operators(py::module& m, const std::string& type_name) {
    // GEMM算子
    py::class_<Gemm<T>>(m, ("Gemm" + type_name).c_str())
        .def(py::init<>())
        .def("set_weight", &Gemm<T>::SetWeight)
        .def("forward", [](Gemm<T>& self, const Tensor<T>& input, Tensor<T>& output) {
            auto status = self.Forward(input, output);
            if (status != StatusCode::SUCCESS) {
                throw std::runtime_error("GEMM forward failed");
            }
        });
    
    // GEMV算子
    py::class_<Gemv<T>>(m, ("Gemv" + type_name).c_str())
        .def(py::init<>())
        .def("set_weight", &Gemv<T>::SetWeight)
        .def("forward", [](Gemv<T>& self, const Tensor<T>& input, Tensor<T>& output) {
            auto status = self.Forward(input, output);
            if (status != StatusCode::SUCCESS) {
                throw std::runtime_error("GEMV forward failed");
            }
        });
    
    // ReLU算子
    py::class_<Relu<T>>(m, ("Relu" + type_name).c_str())
        .def(py::init<>())
        .def("forward", [](Relu<T>& self, const Tensor<T>& input, Tensor<T>& output) {
            auto status = self.Forward(input, output);
            if (status != StatusCode::SUCCESS) {
                throw std::runtime_error("ReLU forward failed");
            }
        });
    
    // Softmax算子
    py::class_<Softmax<T>>(m, ("Softmax" + type_name).c_str())
        .def(py::init<>())
        .def("forward", [](Softmax<T>& self, const Tensor<T>& input, Tensor<T>& output) {
            auto status = self.Forward(input, output);
            if (status != StatusCode::SUCCESS) {
                throw std::runtime_error("Softmax forward failed");
            }
        });
    
    // MatMul算子
    py::class_<MatMul<T>>(m, ("MatMul" + type_name).c_str())
        .def(py::init<>())
        .def("forward", [](MatMul<T>& self, const Tensor<T>& input, Tensor<T>& output) {
            auto status = self.Forward(input, output);
            if (status != StatusCode::SUCCESS) {
                throw std::runtime_error("MatMul forward failed");
            }
        });
}

// JIT配置绑定
void bind_jit_config(py::module& m) {
    py::class_<JITConfig>(m, "JITConfig")
        .def(py::init<>())
        .def_readwrite("enable_jit", &JITConfig::enable_jit)
        .def_readwrite("kernel_type", &JITConfig::kernel_type)
        .def_readwrite("tile_size", &JITConfig::tile_size)
        .def_readwrite("block_size", &JITConfig::block_size)
        .def_readwrite("optimization_level", &JITConfig::optimization_level)
        .def_readwrite("enable_tensor_core", &JITConfig::enable_tensor_core)
        .def_readwrite("enable_tma", &JITConfig::enable_tma)
        .def_readwrite("max_registers", &JITConfig::max_registers)
        .def_readwrite("enable_shared_memory_opt", &JITConfig::enable_shared_memory_opt)
        .def_readwrite("enable_loop_unroll", &JITConfig::enable_loop_unroll)
        .def_readwrite("enable_memory_coalescing", &JITConfig::enable_memory_coalescing)
        .def("__repr__", [](const JITConfig& self) {
            std::stringstream ss;
            ss << "JITConfig(enable_jit=" << (self.enable_jit ? "True" : "False")
               << ", kernel_type='" << self.kernel_type << "'"
               << ", tile_size=" << self.tile_size
               << ", block_size=" << self.block_size
               << ", optimization_level='" << self.optimization_level << "')";
            return ss.str();
        });
    
    py::class_<GlobalJITConfig>(m, "GlobalJITConfig")
        .def(py::init<>())
        .def_readwrite("enable_jit", &GlobalJITConfig::enable_jit)
        .def_readwrite("enable_auto_tuning", &GlobalJITConfig::enable_auto_tuning)
        .def_readwrite("enable_caching", &GlobalJITConfig::enable_caching)
        .def_readwrite("cache_dir", &GlobalJITConfig::cache_dir)
        .def_readwrite("max_cache_size", &GlobalJITConfig::max_cache_size)
        .def_readwrite("compilation_timeout", &GlobalJITConfig::compilation_timeout)
        .def_readwrite("enable_tensor_core", &GlobalJITConfig::enable_tensor_core)
        .def_readwrite("enable_tma", &GlobalJITConfig::enable_tma)
        .def_readwrite("max_compilation_threads", &GlobalJITConfig::max_compilation_threads)
        .def_readwrite("enable_debug", &GlobalJITConfig::enable_debug);
}

// JIT包装器绑定
template<typename T>
void bind_jit_wrapper(py::module& m, const std::string& type_name) {
    py::class_<JITWrapper<Gemm<T>>>(m, ("JITGemm" + type_name).c_str())
        .def(py::init<Gemm<T>&>())
        .def("enable_jit", &JITWrapper<Gemm<T>>::EnableJIT)
        .def("enable_persistent_cache", &JITWrapper<Gemm<T>>::EnablePersistentCache)
        .def("set_persistent_cache_directory", &JITWrapper<Gemm<T>>::SetPersistentCacheDirectory)
        .def("set_jit_config", &JITWrapper<Gemm<T>>::SetJITConfig)
        .def("forward", [](JITWrapper<Gemm<T>>& self, const Tensor<T>& input, Tensor<T>& output) {
            auto status = self.Forward(input, output);
            if (status != StatusCode::SUCCESS) {
                throw std::runtime_error("JIT GEMM forward failed");
            }
        })
        .def("get_performance_profile", &JITWrapper<Gemm<T>>::GetPerformanceProfile)
        .def("is_jit_compiled", &JITWrapper<Gemm<T>>::IsJITCompiled)
        .def("get_last_error", &JITWrapper<Gemm<T>>::GetLastError);
    
    py::class_<JITWrapper<Gemv<T>>>(m, ("JITGemv" + type_name).c_str())
        .def(py::init<Gemv<T>&>())
        .def("enable_jit", &JITWrapper<Gemv<T>>::EnableJIT)
        .def("enable_persistent_cache", &JITWrapper<Gemv<T>>::EnablePersistentCache)
        .def("set_persistent_cache_directory", &JITWrapper<Gemv<T>>::SetPersistentCacheDirectory)
        .def("set_jit_config", &JITWrapper<Gemv<T>>::SetJITConfig)
        .def("forward", [](JITWrapper<Gemv<T>>& self, const Tensor<T>& input, Tensor<T>& output) {
            auto status = self.Forward(input, output);
            if (status != StatusCode::SUCCESS) {
                throw std::runtime_error("JIT GEMV forward failed");
            }
        })
        .def("get_performance_profile", &JITWrapper<Gemv<T>>::GetPerformanceProfile)
        .def("is_jit_compiled", &JITWrapper<Gemv<T>>::IsJITCompiled)
        .def("get_last_error", &JITWrapper<Gemv<T>>::GetLastError);
}

// 性能分析绑定
void bind_performance_profile(py::module& m) {
    py::class_<PerformanceProfile>(m, "PerformanceProfile")
        .def(py::init<>())
        .def_readwrite("execution_time", &PerformanceProfile::execution_time)
        .def_readwrite("kernel_type", &PerformanceProfile::kernel_type)
        .def_readwrite("matrix_size", &PerformanceProfile::matrix_size)
        .def_readwrite("throughput", &PerformanceProfile::throughput)
        .def_readwrite("gflops", &PerformanceProfile::gflops)
        .def("__repr__", [](const PerformanceProfile& self) {
            std::stringstream ss;
            ss << "PerformanceProfile(execution_time=" << self.execution_time
               << "ms, kernel_type='" << self.kernel_type << "'"
               << ", gflops=" << self.gflops << ")";
            return ss.str();
        });
}

// 工具函数绑定
void bind_utility_functions(py::module& m) {
    // 创建Tensor的便捷函数
    m.def("tensor", [](py::array_t<float> arr) {
        return numpy_to_tensor<float>(arr);
    }, "Create a float tensor from numpy array");
    
    m.def("tensor", [](py::array_t<double> arr) {
        return numpy_to_tensor<double>(arr);
    }, "Create a double tensor from numpy array");
    
    m.def("tensor", [](py::array_t<int> arr) {
        return numpy_to_tensor<int>(arr);
    }, "Create an int tensor from numpy array");
    
    // 创建随机Tensor
    m.def("randn", [](const std::vector<size_t>& shape, float mean = 0.0f, float std = 1.0f) {
        Tensor<float> tensor(shape);
        // TODO: 实现随机数生成
        tensor.Fill(mean);
        return tensor;
    }, "Create a random tensor with normal distribution");
    
    m.def("ones", [](const std::vector<size_t>& shape) {
        Tensor<float> tensor(shape);
        tensor.Fill(1.0f);
        return tensor;
    }, "Create a tensor filled with ones");
    
    m.def("zeros", [](const std::vector<size_t>& shape) {
        Tensor<float> tensor(shape);
        tensor.Zero();
        return tensor;
    }, "Create a tensor filled with zeros");
    
    // CUDA设备管理
    m.def("get_device_count", []() {
        int count;
        cudaGetDeviceCount(&count);
        return count;
    }, "Get the number of available CUDA devices");
    
    m.def("set_device", [](int device_id) {
        cudaSetDevice(device_id);
        check_cuda_error();
    }, "Set the current CUDA device");
    
    m.def("get_device", []() {
        int device_id;
        cudaGetDevice(&device_id);
        return device_id;
    }, "Get the current CUDA device");
    
    m.def("synchronize", []() {
        cudaDeviceSynchronize();
        check_cuda_error();
    }, "Synchronize the current CUDA device");
    
    // 内存管理
    m.def("get_memory_info", []() {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        return py::make_tuple(free, total);
    }, "Get CUDA memory information");
    
    m.def("empty_cache", []() {
        cudaDeviceReset();
    }, "Empty CUDA cache and reset device");
}

// 主模块绑定
PYBIND11_MODULE(cuop_core, m) {
    m.doc() = R"pbdoc(
        cuOP Python Bindings
        
        High-performance CUDA operator and memory management library
        with JIT optimization support.
        
        Example:
            >>> import cuop.core as cuop
            >>> import numpy as np
            >>> 
            >>> # Create tensors
            >>> a = cuop.tensor(np.random.randn(1000, 1000).astype(np.float32))
            >>> b = cuop.tensor(np.random.randn(1000, 1000).astype(np.float32))
            >>> c = cuop.zeros([1000, 1000])
            >>> 
            >>> # Use GEMM operator
            >>> gemm = cuop.GemmFloat()
            >>> gemm.set_weight(b)
            >>> gemm.forward(a, c)
            >>> 
            >>> # Convert back to numpy
            >>> result = c.to_numpy()
    )pbdoc";
    
    // 绑定Tensor类
    bind_tensor<float>(m, "Float");
    bind_tensor<double>(m, "Double");
    bind_tensor<int>(m, "Int");
    
    // 绑定基础算子
    bind_basic_operators<float>(m, "Float");
    bind_basic_operators<double>(m, "Double");
    
    // 绑定JIT配置
    bind_jit_config(m);
    
    // 绑定JIT包装器
    bind_jit_wrapper<float>(m, "Float");
    bind_jit_wrapper<double>(m, "Double");
    
    // 绑定性能分析
    bind_performance_profile(m);
    
    // 绑定工具函数
    bind_utility_functions(m);
    
    // 版本信息
    m.attr("__version__") = "0.1.0";
    m.attr("__cuda_enabled__") = true;  // TODO: 动态检测
    
    // 异常类
    py::register_exception<CUDAError>(m, "CUDAError");
    py::register_exception<MemoryError>(m, "MemoryError");
    py::register_exception<CompilationError>(m, "CompilationError");
    py::register_exception<ExecutionError>(m, "ExecutionError");
} 