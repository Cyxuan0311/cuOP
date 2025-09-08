#pragma once

#include "jit_config.hpp"
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <thread>

namespace cu_op_mem {

// 缓存元数据结构
struct CacheMetadata {
    std::string cache_key;                    // 缓存键
    std::string kernel_name;                  // 内核名称
    std::string kernel_code_hash;             // 内核代码哈希
    std::vector<std::string> compile_options; // 编译选项
    std::string ptx_code_hash;                // PTX代码哈希
    std::string cuda_version;                 // CUDA版本
    std::string compute_capability;           // 计算能力
    std::string hardware_info;                // 硬件信息
    std::chrono::system_clock::time_point created_time;    // 创建时间
    std::chrono::system_clock::time_point last_used_time;  // 最后使用时间
    size_t usage_count;                       // 使用次数
    size_t ptx_size;                          // PTX代码大小
    size_t compilation_time_ms;               // 编译时间(毫秒)
    bool is_valid;                            // 是否有效
    
    CacheMetadata() : usage_count(0), ptx_size(0), compilation_time_ms(0), is_valid(true) {}
};

// 持久化缓存统计
struct PersistentCacheStats {
    std::atomic<size_t> total_cached_kernels{0};      // 总缓存内核数
    std::atomic<size_t> disk_cache_hits{0};           // 磁盘缓存命中数
    std::atomic<size_t> disk_cache_misses{0};         // 磁盘缓存未命中数
    std::atomic<size_t> total_disk_cache_size{0};     // 总磁盘缓存大小
    std::atomic<size_t> total_saved_compilation_time{0}; // 节省的总编译时间
    std::atomic<size_t> cache_evictions{0};           // 缓存驱逐次数
    std::atomic<size_t> cache_corruptions{0};         // 缓存损坏次数
    
    // 默认构造函数
    PersistentCacheStats() = default;
    
    void Reset() {
        total_cached_kernels = 0;
        disk_cache_hits = 0;
        disk_cache_misses = 0;
        total_disk_cache_size = 0;
        total_saved_compilation_time = 0;
        cache_evictions = 0;
        cache_corruptions = 0;
    }
    
    // 移动构造函数
    PersistentCacheStats(PersistentCacheStats&& other) noexcept
        : total_cached_kernels(other.total_cached_kernels.load()),
          disk_cache_hits(other.disk_cache_hits.load()),
          disk_cache_misses(other.disk_cache_misses.load()),
          total_disk_cache_size(other.total_disk_cache_size.load()),
          total_saved_compilation_time(other.total_saved_compilation_time.load()),
          cache_evictions(other.cache_evictions.load()),
          cache_corruptions(other.cache_corruptions.load()) {}
    
    // 移动赋值操作符
    PersistentCacheStats& operator=(PersistentCacheStats&& other) noexcept {
        if (this != &other) {
            total_cached_kernels = other.total_cached_kernels.load();
            disk_cache_hits = other.disk_cache_hits.load();
            disk_cache_misses = other.disk_cache_misses.load();
            total_disk_cache_size = other.total_disk_cache_size.load();
            total_saved_compilation_time = other.total_saved_compilation_time.load();
            cache_evictions = other.cache_evictions.load();
            cache_corruptions = other.cache_corruptions.load();
        }
        return *this;
    }
    
    // 删除复制构造函数和复制赋值操作符
    PersistentCacheStats(const PersistentCacheStats&) = delete;
    PersistentCacheStats& operator=(const PersistentCacheStats&) = delete;
};

// 缓存策略配置
struct CachePolicy {
    size_t max_disk_cache_size = 10ULL * 1024 * 1024 * 1024;  // 最大磁盘缓存大小(10GB)
    size_t max_cached_kernels = 10000;                         // 最大缓存内核数
    std::chrono::hours cache_expiration_time{168};             // 缓存过期时间(7天)
    std::chrono::hours cleanup_interval{24};                   // 清理间隔(24小时)
    bool enable_compression = true;                            // 启用压缩
    bool enable_checksum = true;                               // 启用校验和
    bool enable_versioning = true;                             // 启用版本控制
    float eviction_threshold = 0.8f;                           // 驱逐阈值(80%)
    
    CachePolicy() = default;
};

// 持久化缓存管理器
class JITPersistentCache {
public:
    explicit JITPersistentCache(const std::string& cache_dir, const CachePolicy& policy = CachePolicy{});
    ~JITPersistentCache();
    
    // 禁用拷贝
    JITPersistentCache(const JITPersistentCache&) = delete;
    JITPersistentCache& operator=(const JITPersistentCache&) = delete;
    
    // 缓存管理
    bool SaveKernel(const std::string& cache_key, 
                   const std::string& kernel_name,
                   const std::string& kernel_code,
                   const std::vector<std::string>& compile_options,
                   const std::string& ptx_code,
                   double compilation_time_ms);
    
    bool LoadKernel(const std::string& cache_key,
                   std::string& kernel_name,
                   std::vector<std::string>& compile_options,
                   std::string& ptx_code);
    
    bool IsKernelCached(const std::string& cache_key) const;
    void RemoveKernel(const std::string& cache_key);
    
    // 缓存查询
    std::vector<CacheMetadata> GetAllCachedKernels() const;
    CacheMetadata GetKernelMetadata(const std::string& cache_key) const;
    std::vector<std::string> SearchKernels(const std::string& pattern) const;
    
    // 缓存维护
    void CleanupExpiredCache();
    void CleanupBySize(size_t target_size);
    void ValidateCacheIntegrity();
    void CompactCache();
    
    // 统计信息
    PersistentCacheStats GetStats() const;
    void ResetStats();
    void PrintStats() const;
    
    // 配置管理
    void SetCachePolicy(const CachePolicy& policy);
    CachePolicy GetCachePolicy() const;
    
    // 缓存导入/导出
    bool ExportCache(const std::string& export_path) const;
    bool ImportCache(const std::string& import_path);
    
    // 版本管理
    bool IsCompatibleWithCurrentEnvironment() const;
    std::string GetCacheVersion() const;
    void MigrateCacheIfNeeded();
    
    // 性能优化
    void PreloadFrequentlyUsedKernels();
    void OptimizeCacheLayout();
    
private:
    // 内部实现
    std::string GenerateCacheFilePath(const std::string& cache_key) const;
    std::string GenerateMetadataFilePath(const std::string& cache_key) const;
    std::string GenerateKernelCodeHash(const std::string& kernel_code) const;
    std::string GeneratePTXHash(const std::string& ptx_code) const;
    std::string GetCurrentEnvironmentSignature() const;
    
    bool SaveMetadata(const std::string& cache_key, const CacheMetadata& metadata);
    bool LoadMetadata(const std::string& cache_key, CacheMetadata& metadata) const;
    bool SavePTXCode(const std::string& cache_key, const std::string& ptx_code);
    bool LoadPTXCode(const std::string& cache_key, std::string& ptx_code);
    
    void UpdateUsageStatistics(const std::string& cache_key);
    void PerformCacheMaintenance();
    void StartMaintenanceThread();
    void StopMaintenanceThread();
    void ScanExistingCache();
    void SaveStatistics();
    
    // 压缩相关
    std::vector<uint8_t> CompressData(const std::string& data) const;
    bool DecompressData(const std::vector<uint8_t>& compressed_data, std::string& data) const;
    
    // 校验和
    std::string CalculateChecksum(const std::string& data) const;
    bool VerifyChecksum(const std::string& data, const std::string& expected_checksum) const;
    
    // 成员变量
    std::string cache_dir_;
    CachePolicy policy_;
    PersistentCacheStats stats_;
    
    // 内存缓存(快速访问)
    mutable std::unordered_map<std::string, CacheMetadata> metadata_cache_;
    mutable std::unordered_map<std::string, std::string> ptx_cache_;
    
    // 同步
    mutable std::mutex cache_mutex_;
    mutable std::mutex stats_mutex_;
    
    // 维护线程
    std::thread maintenance_thread_;
    std::atomic<bool> stop_maintenance_{false};
    
    // 文件系统监控
    std::chrono::system_clock::time_point last_maintenance_time_;
    std::filesystem::file_time_type last_cache_scan_time_;
};

// 全局持久化缓存管理器
class GlobalPersistentCacheManager {
public:
    static GlobalPersistentCacheManager& Instance();
    
    // 初始化
    bool Initialize(const std::string& cache_dir, const CachePolicy& policy = CachePolicy{});
    void Cleanup();
    
    // 缓存操作
    bool SaveKernel(const std::string& cache_key, 
                   const std::string& kernel_name,
                   const std::string& kernel_code,
                   const std::vector<std::string>& compile_options,
                   const std::string& ptx_code,
                   double compilation_time_ms);
    
    bool LoadKernel(const std::string& cache_key,
                   std::string& kernel_name,
                   std::vector<std::string>& compile_options,
                   std::string& ptx_code);
    
    bool IsKernelCached(const std::string& cache_key) const;
    
    // 获取内核元数据
    CacheMetadata GetKernelMetadata(const std::string& cache_key) const;
    
    // 统计和配置
    PersistentCacheStats GetStats() const;
    CachePolicy GetCachePolicy() const;
    void SetCachePolicy(const CachePolicy& policy);
    
    // 维护操作
    void CleanupExpiredCache();
    void ValidateCacheIntegrity();
    
private:
    GlobalPersistentCacheManager() = default;
    ~GlobalPersistentCacheManager() = default;
    
    std::unique_ptr<JITPersistentCache> cache_manager_;
    std::string cache_dir_;
    CachePolicy policy_;
    bool initialized_ = false;
    mutable std::mutex mutex_;
};

// 宏定义：简化缓存使用
#define JIT_PERSISTENT_CACHE_SAVE(key, name, code, options, ptx, time) \
    GlobalPersistentCacheManager::Instance().SaveKernel(key, name, code, options, ptx, time)

#define JIT_PERSISTENT_CACHE_LOAD(key, name, options, ptx) \
    GlobalPersistentCacheManager::Instance().LoadKernel(key, name, options, ptx)

#define JIT_PERSISTENT_CACHE_CHECK(key) \
    GlobalPersistentCacheManager::Instance().IsKernelCached(key)

} // namespace cu_op_mem 