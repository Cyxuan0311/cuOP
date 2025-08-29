#include "jit/Blas/blas_jit_plugins.hpp"
#include "jit/ijit_plugin.hpp"
#include <glog/logging.h>

namespace cu_op_mem {

BlasJITPluginManager& BlasJITPluginManager::Instance() {
    static BlasJITPluginManager instance;
    return instance;
}

void BlasJITPluginManager::RegisterAllPlugins() {
    // 注册GEMM插件
    plugin_factories_["gemm"] = []() -> std::unique_ptr<IJITPlugin> {
        return std::make_unique<GemmJITPlugin>();
    };
    
    plugin_factories_["Gemm"] = []() -> std::unique_ptr<IJITPlugin> {
        return std::make_unique<GemmJITPlugin>();
    };
    
    // 注册GEMV插件
    plugin_factories_["gemv"] = []() -> std::unique_ptr<IJITPlugin> {
        return std::make_unique<GemvJITPlugin>();
    };
    
    plugin_factories_["Gemv"] = []() -> std::unique_ptr<IJITPlugin> {
        return std::make_unique<GemvJITPlugin>();
    };
    
    // 注册TRSM插件
    plugin_factories_["trsm"] = []() -> std::unique_ptr<IJITPlugin> {
        return std::make_unique<TrsmJITPlugin>();
    };
    
    plugin_factories_["Trsm"] = []() -> std::unique_ptr<IJITPlugin> {
        return std::make_unique<TrsmJITPlugin>();
    };
    
    // 未来可以在这里添加更多BLAS插件
    // 例如：GEMM_BATCHED、SYMM等
    
    VLOG(1) << "Registered " << plugin_factories_.size() << " BLAS JIT plugins";
}

std::vector<std::string> BlasJITPluginManager::GetSupportedOperators() const {
    std::vector<std::string> operators;
    for (const auto& pair : plugin_factories_) {
        operators.push_back(pair.first);
    }
    return operators;
}

bool BlasJITPluginManager::SupportsOperator(const std::string& op_name) const {
    return plugin_factories_.find(op_name) != plugin_factories_.end();
}

std::unique_ptr<IJITPlugin> BlasJITPluginManager::CreatePlugin(const std::string& op_name) {
    auto it = plugin_factories_.find(op_name);
    if (it != plugin_factories_.end()) {
        return it->second();
    }
    
    LOG(WARNING) << "BLAS JIT plugin not found for operator: " << op_name;
    return nullptr;
}

} // namespace cu_op_mem 