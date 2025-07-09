#pragma once

namespace cu_op_mem {

enum class StatusCode {
    SUCCESS = 0,
    CUDA_ERROR,
    SHAPE_MISMATCH,
    UNSUPPORTED_TYPE,
    UNKNOWN_ERROR
};

// 可选：将错误码转为字符串
inline const char* StatusCodeToString(StatusCode code) {
    switch (code) {
        case StatusCode::SUCCESS: return "SUCCESS";
        case StatusCode::CUDA_ERROR: return "CUDA_ERROR";
        case StatusCode::SHAPE_MISMATCH: return "SHAPE_MISMATCH";
        case StatusCode::UNSUPPORTED_TYPE: return "UNSUPPORTED_TYPE";
        case StatusCode::UNKNOWN_ERROR: return "UNKNOWN_ERROR";
        default: return "UNKNOWN";
    }
}

} // namespace cu_op_mem