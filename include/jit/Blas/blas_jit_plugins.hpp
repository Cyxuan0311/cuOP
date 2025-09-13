#pragma once

// BLAS JIT插件统一入口
// 包含所有BLAS相关的JIT插件实现

#include "jit/ijit_plugin.hpp"
#include "jit/Blas/gemm_jit_plugin.hpp"
#include "jit/Blas/gemv_jit_plugin.hpp"
#include "jit/Blas/trsm_jit_plugin.hpp"
#include <unordered_map>
#include <functional>
#include <memory>
#include <vector>
#include <string>

namespace cu_op_mem {

// BLAS JIT插件管理器
class BlasJITPluginManager {
public:
    static BlasJITPluginManager& Instance();
    
    // 注册所有BLAS插件
    void RegisterAllPlugins();
    
    // 获取支持的算子列表
    std::vector<std::string> GetSupportedOperators() const;
    
    // 检查是否支持某个算子
    bool SupportsOperator(const std::string& op_name) const;
    
    // 创建插件
    std::unique_ptr<IJITPlugin> CreatePlugin(const std::string& op_name);
    
private:
    BlasJITPluginManager() = default;
    ~BlasJITPluginManager() = default;
    BlasJITPluginManager(const BlasJITPluginManager&) = delete;
    BlasJITPluginManager& operator=(const BlasJITPluginManager&) = delete;
    
    std::unordered_map<std::string, std::function<std::unique_ptr<IJITPlugin>()>> plugin_factories_;
};

// 便捷函数
inline void RegisterBlasJITPlugins() {
    BlasJITPluginManager::Instance().RegisterAllPlugins();
}

inline std::vector<std::string> GetSupportedBlasOperators() {
    return BlasJITPluginManager::Instance().GetSupportedOperators();
}

inline bool IsBlasOperatorSupported(const std::string& op_name) {
    return BlasJITPluginManager::Instance().SupportsOperator(op_name);
}

inline std::unique_ptr<IJITPlugin> CreateBlasJITPlugin(const std::string& op_name) {
    return BlasJITPluginManager::Instance().CreatePlugin(op_name);
}

} // namespace cu_op_mem 