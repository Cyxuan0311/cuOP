#pragma once

#include "jit_config.hpp"
#include "jit/ijit_plugin.hpp"
#include <unordered_map>
#include <mutex>
#include <string>

namespace cu_op_mem {

// 全局JIT管理器
class GlobalJITManager {
public:
    static GlobalJITManager& Instance();
    
    // 全局配置管理
    void SetGlobalConfig(const GlobalJITConfig& config);
    GlobalJITConfig GetGlobalConfig() const;
    
    // 算子配置管理
    void SetOperatorConfig(const std::string& op_name, const JITConfig& config);
    JITConfig GetOperatorConfig(const std::string& op_name) const;
    
    // 系统管理
    StatusCode Initialize();
    void Cleanup();
    
    // 统计信息
    JITStatistics GetStatistics() const;
    void ResetStatistics();
    
    // 缓存管理
    void ClearAllCaches();
    void SaveCacheToFile(const std::string& file_path);
    void LoadCacheFromFile(const std::string& file_path);
    
private:
    GlobalJITManager() = default;
    ~GlobalJITManager() = default;
    GlobalJITManager(const GlobalJITManager&) = delete;
    GlobalJITManager& operator=(const GlobalJITManager&) = delete;
    
    GlobalJITConfig global_config_;
    std::unordered_map<std::string, JITConfig> operator_configs_;
    JITStatistics global_statistics_;
    mutable std::mutex config_mutex_;
    mutable std::mutex stats_mutex_;
};

} // namespace cu_op_mem
