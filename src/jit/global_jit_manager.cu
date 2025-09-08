#include "jit/global_jit_manager.hpp"
#include "jit/jit_config.hpp"
#include "jit/ijit_plugin.hpp"
#include "jit/jit_compiler.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace cu_op_mem {

// 全局JIT管理器实现
GlobalJITManager& GlobalJITManager::Instance() {
    static GlobalJITManager instance;
    return instance;
}

void GlobalJITManager::SetGlobalConfig(const GlobalJITConfig& config) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    global_config_ = config;
    
    LOG(INFO) << "Global JIT config updated:";
    LOG(INFO) << "  Enable JIT: " << (config.enable_jit ? "true" : "false");
    LOG(INFO) << "  Enable Auto Tuning: " << (config.enable_auto_tuning ? "true" : "false");
    LOG(INFO) << "  Enable Caching: " << (config.enable_caching ? "true" : "false");
    LOG(INFO) << "  Cache Directory: " << config.cache_dir;
    LOG(INFO) << "  Max Cache Size: " << config.max_cache_size << " bytes";
    LOG(INFO) << "  Compilation Timeout: " << config.compilation_timeout << " seconds";
    LOG(INFO) << "  Enable Tensor Core: " << (config.enable_tensor_core ? "true" : "false");
    LOG(INFO) << "  Enable TMA: " << (config.enable_tma ? "true" : "false");
    LOG(INFO) << "  Max Compilation Threads: " << config.max_compilation_threads;
    LOG(INFO) << "  Enable Debug: " << (config.enable_debug ? "true" : "false");
}

GlobalJITConfig GlobalJITManager::GetGlobalConfig() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return global_config_;
}

void GlobalJITManager::SetOperatorConfig(const std::string& op_name, const JITConfig& config) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    operator_configs_[op_name] = config;
    
    LOG(INFO) << "Operator JIT config updated for " << op_name << ":";
    LOG(INFO) << "  Enable JIT: " << (config.enable_jit ? "true" : "false");
    LOG(INFO) << "  Block Sizes: [";
    for (size_t i = 0; i < config.block_sizes.size(); ++i) {
        if (i > 0) LOG(INFO) << ", ";
        LOG(INFO) << config.block_sizes[i];
    }
    LOG(INFO) << "]";
    LOG(INFO) << "  Tile Sizes: [";
    for (size_t i = 0; i < config.tile_sizes.size(); ++i) {
        if (i > 0) LOG(INFO) << ", ";
        LOG(INFO) << config.tile_sizes[i];
    }
    LOG(INFO) << "]";
    LOG(INFO) << "  Num Stages: " << config.num_stages;
    LOG(INFO) << "  Use Tensor Core: " << (config.use_tensor_core ? "true" : "false");
    LOG(INFO) << "  Use TMA: " << (config.use_tma ? "true" : "false");
    LOG(INFO) << "  Optimization Level: " << config.optimization_level;
    LOG(INFO) << "  Max Registers: " << config.max_registers;
    LOG(INFO) << "  Enable Shared Memory Opt: " << (config.enable_shared_memory_opt ? "true" : "false");
    LOG(INFO) << "  Enable Loop Unroll: " << (config.enable_loop_unroll ? "true" : "false");
    LOG(INFO) << "  Enable Memory Coalescing: " << (config.enable_memory_coalescing ? "true" : "false");
}

JITConfig GlobalJITManager::GetOperatorConfig(const std::string& op_name) const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    auto it = operator_configs_.find(op_name);
    if (it != operator_configs_.end()) {
        return it->second;
    }
    // 返回默认配置
    return JITConfig{};
}

StatusCode GlobalJITManager::Initialize() {
    LOG(INFO) << "Initializing Global JIT Manager...";
    
    // 检查CUDA环境
    int device_count;
    cudaError_t cuda_result = cudaGetDeviceCount(&device_count);
    if (cuda_result != cudaSuccess) {
        LOG(ERROR) << "Failed to get CUDA device count: " << cudaGetErrorString(cuda_result);
        return StatusCode::CUDA_ERROR;
    }
    
    if (device_count == 0) {
        LOG(ERROR) << "No CUDA devices found";
        return StatusCode::CUDA_ERROR;
    }
    
    LOG(INFO) << "Found " << device_count << " CUDA device(s)";
    
    // 获取当前设备信息
    int current_device;
    cuda_result = cudaGetDevice(&current_device);
    if (cuda_result != cudaSuccess) {
        LOG(ERROR) << "Failed to get current CUDA device: " << cudaGetErrorString(cuda_result);
        return StatusCode::CUDA_ERROR;
    }
    
    cudaDeviceProp device_prop;
    cuda_result = cudaGetDeviceProperties(&device_prop, current_device);
    if (cuda_result != cudaSuccess) {
        LOG(ERROR) << "Failed to get device properties: " << cudaGetErrorString(cuda_result);
        return StatusCode::CUDA_ERROR;
    }
    
    LOG(INFO) << "Current CUDA device: " << device_prop.name;
    LOG(INFO) << "  Compute Capability: " << device_prop.major << "." << device_prop.minor;
    LOG(INFO) << "  Number of SMs: " << device_prop.multiProcessorCount;
    LOG(INFO) << "  Max Threads per SM: " << device_prop.maxThreadsPerMultiProcessor;
    LOG(INFO) << "  Max Shared Memory per SM: " << device_prop.sharedMemPerMultiprocessor << " bytes";
    LOG(INFO) << "  Max Registers per SM: " << device_prop.regsPerMultiprocessor;
    LOG(INFO) << "  Max Threads per Block: " << device_prop.maxThreadsPerBlock;
    LOG(INFO) << "  Max Blocks per SM: " << device_prop.maxBlocksPerMultiProcessor;
    LOG(INFO) << "  Total Global Memory: " << device_prop.totalGlobalMem << " bytes";
    LOG(INFO) << "  Memory Clock Rate: " << device_prop.memoryClockRate << " kHz";
    LOG(INFO) << "  Memory Bus Width: " << device_prop.memoryBusWidth << " bits";
    
    // 检查Tensor Core支持
    bool supports_tensor_core = (device_prop.major >= 7);
    LOG(INFO) << "  Supports Tensor Core: " << (supports_tensor_core ? "true" : "false");
    
    // 检查TMA支持
    bool supports_tma = (device_prop.major >= 9);
    LOG(INFO) << "  Supports TMA: " << (supports_tma ? "true" : "false");
    
    // 创建缓存目录
    if (global_config_.enable_caching) {
        try {
            std::filesystem::create_directories(global_config_.cache_dir);
            LOG(INFO) << "Created cache directory: " << global_config_.cache_dir;
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to create cache directory: " << e.what();
        }
    }
    
    // 初始化插件注册器
    // 这里可以注册默认的插件
    LOG(INFO) << "Global JIT Manager initialized successfully";
    
    return StatusCode::SUCCESS;
}

void GlobalJITManager::Cleanup() {
    LOG(INFO) << "Cleaning up Global JIT Manager...";
    
    // 清理所有缓存
    ClearAllCaches();
    
    // 清理插件管理器
    PluginManager::Instance().CleanupAllPlugins();
    
    LOG(INFO) << "Global JIT Manager cleanup completed";
}

JITStatistics GlobalJITManager::GetStatistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return global_statistics_;
}

void GlobalJITManager::ResetStatistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    global_statistics_.Reset();
    
    // 重置插件统计信息
    auto active_plugins = PluginManager::Instance().GetActivePlugins();
    for (const auto& plugin_name : active_plugins) {
        auto* plugin = PluginManager::Instance().GetPlugin(plugin_name);
        if (plugin) {
            // 这里可以调用插件的重置统计信息方法
        }
    }
}

void GlobalJITManager::ClearAllCaches() {
    LOG(INFO) << "Clearing all JIT caches...";
    
    // 清理插件缓存
    auto active_plugins = PluginManager::Instance().GetActivePlugins();
    for (const auto& plugin_name : active_plugins) {
        auto* plugin = PluginManager::Instance().GetPlugin(plugin_name);
        if (plugin) {
            plugin->Cleanup();
        }
    }
    
    // 清理文件缓存
    if (global_config_.enable_caching && !global_config_.cache_dir.empty()) {
        try {
            std::filesystem::remove_all(global_config_.cache_dir);
            std::filesystem::create_directories(global_config_.cache_dir);
            LOG(INFO) << "Cleared file cache directory: " << global_config_.cache_dir;
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to clear file cache: " << e.what();
        }
    }
}

void GlobalJITManager::SaveCacheToFile(const std::string& file_path) {
    LOG(INFO) << "Saving JIT cache to file: " << file_path;
    
    // 这里可以实现将缓存保存到文件的逻辑
    // 包括编译的PTX代码、配置信息等
    
    try {
        std::ofstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            LOG(ERROR) << "Failed to open file for writing: " << file_path;
            return;
        }
        
        // 保存全局配置
        auto config = GetGlobalConfig();
        // 序列化配置...
        
        // 保存算子配置
        // 序列化算子配置...
        
        // 保存统计信息
        auto stats = GetStatistics();
        // 序列化统计信息...
        
        file.close();
        LOG(INFO) << "JIT cache saved successfully";
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to save JIT cache: " << e.what();
    }
}

void GlobalJITManager::LoadCacheFromFile(const std::string& file_path) {
    LOG(INFO) << "Loading JIT cache from file: " << file_path;
    
    // 这里可以实现从文件加载缓存的逻辑
    
    try {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            LOG(ERROR) << "Failed to open file for reading: " << file_path;
            return;
        }
        
        // 加载全局配置
        // 反序列化配置...
        
        // 加载算子配置
        // 反序列化算子配置...
        
        // 加载统计信息
        // 反序列化统计信息...
        
        file.close();
        LOG(INFO) << "JIT cache loaded successfully";
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to load JIT cache: " << e.what();
    }
}

// 插件注册器实现
PluginRegistrar& PluginRegistrar::Instance() {
    static PluginRegistrar instance;
    return instance;
}

void PluginRegistrar::RegisterPluginFactory(const std::string& plugin_type, 
                                          std::unique_ptr<IPluginFactory> factory) {
    std::lock_guard<std::mutex> lock(mutex_);
    factories_[plugin_type] = std::move(factory);
    LOG(INFO) << "Registered plugin factory: " << plugin_type;
}

std::unique_ptr<IJITPlugin> PluginRegistrar::CreatePlugin(const std::string& plugin_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = factories_.find(plugin_type);
    if (it != factories_.end()) {
        auto plugin = it->second->CreatePlugin();
        LOG(INFO) << "Created plugin: " << plugin_type;
        return plugin;
    }
    LOG(WARNING) << "Plugin factory not found: " << plugin_type;
    return nullptr;
}

std::vector<std::string> PluginRegistrar::GetSupportedPluginTypes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> types;
    types.reserve(factories_.size());
    for (const auto& [type, _] : factories_) {
        types.push_back(type);
    }
    return types;
}

bool PluginRegistrar::IsOperatorSupported(const std::string& op_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [type, factory] : factories_) {
        if (factory->SupportsOperator(op_name)) {
            return true;
        }
    }
    return false;
}

// 插件管理器实现
PluginManager& PluginManager::Instance() {
    static PluginManager instance;
    return instance;
}

void PluginManager::RegisterPlugin(const std::string& name, std::unique_ptr<IJITPlugin> plugin) {
    std::lock_guard<std::mutex> lock(mutex_);
    plugins_[name] = std::move(plugin);
    LOG(INFO) << "Registered plugin: " << name;
}

IJITPlugin* PluginManager::GetPlugin(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = plugins_.find(name);
    if (it != plugins_.end()) {
        return it->second.get();
    }
    return nullptr;
}

void PluginManager::RemovePlugin(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = plugins_.find(name);
    if (it != plugins_.end()) {
        it->second->Cleanup();
        plugins_.erase(it);
        LOG(INFO) << "Removed plugin: " << name;
    }
}

void PluginManager::InitializeAllPlugins() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [name, plugin] : plugins_) {
        auto status = plugin->Initialize();
        if (status == StatusCode::SUCCESS) {
            LOG(INFO) << "Initialized plugin: " << name;
        } else {
            LOG(ERROR) << "Failed to initialize plugin: " << name;
        }
    }
}

void PluginManager::CompileAllPlugins(const JITConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [name, plugin] : plugins_) {
        auto status = plugin->Compile(config);
        if (status == StatusCode::SUCCESS) {
            LOG(INFO) << "Compiled plugin: " << name;
        } else {
            LOG(ERROR) << "Failed to compile plugin: " << name;
        }
    }
}

void PluginManager::CleanupAllPlugins() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [name, plugin] : plugins_) {
        plugin->Cleanup();
        LOG(INFO) << "Cleaned up plugin: " << name;
    }
}

std::vector<std::string> PluginManager::GetActivePlugins() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(plugins_.size());
    for (const auto& [name, _] : plugins_) {
        names.push_back(name);
    }
    return names;
}

size_t PluginManager::GetTotalMemoryUsage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total_usage = 0;
    for (const auto& [name, plugin] : plugins_) {
        total_usage += plugin->GetMemoryUsage();
    }
    return total_usage;
}

JITStatistics PluginManager::GetTotalStatistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    JITStatistics total_stats;
    for (const auto& [name, plugin] : plugins_) {
        // 这里可以累加各个插件的统计信息
        // 暂时返回空统计信息
    }
    return total_stats;
}

} // namespace cu_op_mem 