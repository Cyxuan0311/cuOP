#include "jit/jit_persistent_cache.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <zlib.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <cstring>
#include <openssl/sha.h>

namespace cu_op_mem {

// ==================== JITPersistentCache 实现 ====================

JITPersistentCache::JITPersistentCache(const std::string& cache_dir, const CachePolicy& policy)
    : cache_dir_(cache_dir), policy_(policy), last_maintenance_time_(std::chrono::system_clock::now()) {
    
    // 创建缓存目录
    try {
        std::filesystem::create_directories(cache_dir_);
        std::filesystem::create_directories(cache_dir_ + "/kernels");
        std::filesystem::create_directories(cache_dir_ + "/metadata");
        std::filesystem::create_directories(cache_dir_ + "/temp");
        
        LOG(INFO) << "JIT Persistent Cache initialized at: " << cache_dir_;
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to create cache directories: " << e.what();
    }
    
    // 启动维护线程
    StartMaintenanceThread();
    
    // 扫描现有缓存
    ScanExistingCache();
}

JITPersistentCache::~JITPersistentCache() {
    StopMaintenanceThread();
    
    // 保存统计信息
    SaveStatistics();
    
    LOG(INFO) << "JIT Persistent Cache destroyed";
}

bool JITPersistentCache::SaveKernel(const std::string& cache_key,
                                   const std::string& kernel_name,
                                   const std::string& kernel_code,
                                   const std::vector<std::string>& compile_options,
                                   const std::string& ptx_code,
                                   double compilation_time_ms) {
    try {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        
        // 检查缓存大小限制
        if (stats_.total_cached_kernels.load() >= policy_.max_cached_kernels) {
            CleanupBySize(policy_.max_disk_cache_size * 0.8); // 清理到80%
        }
        
        // 创建缓存元数据
        CacheMetadata metadata;
        metadata.cache_key = cache_key;
        metadata.kernel_name = kernel_name;
        metadata.kernel_code_hash = GenerateKernelCodeHash(kernel_code);
        metadata.compile_options = compile_options;
        metadata.ptx_code_hash = GeneratePTXHash(ptx_code);
        metadata.cuda_version = GetCurrentEnvironmentSignature();
        metadata.compute_capability = GetCurrentEnvironmentSignature();
        metadata.hardware_info = GetCurrentEnvironmentSignature();
        metadata.created_time = std::chrono::system_clock::now();
        metadata.last_used_time = metadata.created_time;
        metadata.usage_count = 1;
        metadata.ptx_size = ptx_code.size();
        metadata.compilation_time_ms = static_cast<size_t>(compilation_time_ms);
        metadata.is_valid = true;
        
        // 保存元数据
        if (!SaveMetadata(cache_key, metadata)) {
            LOG(ERROR) << "Failed to save metadata for kernel: " << cache_key;
            return false;
        }
        
        // 保存PTX代码
        if (!SavePTXCode(cache_key, ptx_code)) {
            LOG(ERROR) << "Failed to save PTX code for kernel: " << cache_key;
            RemoveKernel(cache_key);
            return false;
        }
        
        // 更新内存缓存
        metadata_cache_[cache_key] = metadata;
        ptx_cache_[cache_key] = ptx_code;
        
        // 更新统计信息
        stats_.total_cached_kernels.fetch_add(1);
        stats_.total_disk_cache_size.fetch_add(ptx_code.size());
        stats_.total_saved_compilation_time.fetch_add(static_cast<size_t>(compilation_time_ms));
        
        VLOG(1) << "Successfully cached kernel: " << cache_key 
                << " (PTX size: " << ptx_code.size() << " bytes)";
        
        return true;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Exception while saving kernel: " << e.what();
        return false;
    }
}

bool JITPersistentCache::LoadKernel(const std::string& cache_key,
                                   std::string& kernel_name,
                                   std::vector<std::string>& compile_options,
                                   std::string& ptx_code) {
    try {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        
        // 首先检查内存缓存
        auto metadata_it = metadata_cache_.find(cache_key);
        auto ptx_it = ptx_cache_.find(cache_key);
        
        if (metadata_it != metadata_cache_.end() && ptx_it != ptx_cache_.end()) {
            // 内存缓存命中
            const auto& metadata = metadata_it->second;
            kernel_name = metadata.kernel_name;
            compile_options = metadata.compile_options;
            ptx_code = ptx_it->second;
            
            // 更新使用统计
            UpdateUsageStatistics(cache_key);
            return true;
        }
        
        // 从磁盘加载
        CacheMetadata metadata;
        if (!LoadMetadata(cache_key, metadata)) {
            stats_.disk_cache_misses.fetch_add(1);
            return false;
        }
        
        // 验证环境兼容性
        if (!IsCompatibleWithCurrentEnvironment()) {
            LOG(WARNING) << "Cache environment mismatch for kernel: " << cache_key;
            RemoveKernel(cache_key);
            stats_.disk_cache_misses.fetch_add(1);
            return false;
        }
        
        // 加载PTX代码
        if (!LoadPTXCode(cache_key, ptx_code)) {
            LOG(ERROR) << "Failed to load PTX code for kernel: " << cache_key;
            RemoveKernel(cache_key);
            stats_.disk_cache_misses.fetch_add(1);
            return false;
        }
        
        // 验证PTX代码完整性
        if (metadata.ptx_code_hash != GeneratePTXHash(ptx_code)) {
            LOG(ERROR) << "PTX code corruption detected for kernel: " << cache_key;
            RemoveKernel(cache_key);
            stats_.cache_corruptions.fetch_add(1);
            stats_.disk_cache_misses.fetch_add(1);
            return false;
        }
        
        // 更新内存缓存
        metadata_cache_[cache_key] = metadata;
        ptx_cache_[cache_key] = ptx_code;
        
        // 更新统计信息
        kernel_name = metadata.kernel_name;
        compile_options = metadata.compile_options;
        stats_.disk_cache_hits.fetch_add(1);
        
        VLOG(1) << "Successfully loaded kernel from disk: " << cache_key;
        return true;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Exception while loading kernel: " << e.what();
        stats_.disk_cache_misses.fetch_add(1);
        return false;
    }
}

bool JITPersistentCache::IsKernelCached(const std::string& cache_key) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    // 检查内存缓存
    if (metadata_cache_.find(cache_key) != metadata_cache_.end()) {
        return true;
    }
    
    // 检查磁盘缓存
    std::string metadata_path = GenerateMetadataFilePath(cache_key);
    return std::filesystem::exists(metadata_path);
}

void JITPersistentCache::RemoveKernel(const std::string& cache_key) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    // 从内存缓存移除
    auto metadata_it = metadata_cache_.find(cache_key);
    if (metadata_it != metadata_cache_.end()) {
        stats_.total_disk_cache_size.fetch_sub(metadata_it->second.ptx_size);
        metadata_cache_.erase(metadata_it);
    }
    
    auto ptx_it = ptx_cache_.find(cache_key);
    if (ptx_it != ptx_cache_.end()) {
        ptx_cache_.erase(ptx_it);
    }
    
    // 从磁盘移除
    try {
        std::string metadata_path = GenerateMetadataFilePath(cache_key);
        std::string ptx_path = GenerateCacheFilePath(cache_key);
        
        if (std::filesystem::exists(metadata_path)) {
            std::filesystem::remove(metadata_path);
        }
        if (std::filesystem::exists(ptx_path)) {
            std::filesystem::remove(ptx_path);
        }
        
        stats_.total_cached_kernels.fetch_sub(1);
        VLOG(1) << "Removed kernel from cache: " << cache_key;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to remove kernel from disk: " << e.what();
    }
}

std::vector<CacheMetadata> JITPersistentCache::GetAllCachedKernels() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::vector<CacheMetadata> kernels;
    kernels.reserve(metadata_cache_.size());
    
    for (const auto& [key, metadata] : metadata_cache_) {
        kernels.push_back(metadata);
    }
    
    return kernels;
}

CacheMetadata JITPersistentCache::GetKernelMetadata(const std::string& cache_key) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    auto it = metadata_cache_.find(cache_key);
    if (it != metadata_cache_.end()) {
        return it->second;
    }
    
    // 尝试从磁盘加载
    CacheMetadata metadata;
    if (LoadMetadata(cache_key, metadata)) {
        return metadata;
    }
    
    return CacheMetadata{}; // 返回空元数据
}

void JITPersistentCache::CleanupExpiredCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    auto now = std::chrono::system_clock::now();
    std::vector<std::string> expired_keys;
    
    for (const auto& [key, metadata] : metadata_cache_) {
        if (now - metadata.last_used_time > policy_.cache_expiration_time) {
            expired_keys.push_back(key);
        }
    }
    
    for (const auto& key : expired_keys) {
        RemoveKernel(key);
        stats_.cache_evictions.fetch_add(1);
    }
    
    if (!expired_keys.empty()) {
        LOG(INFO) << "Cleaned up " << expired_keys.size() << " expired kernels";
    }
}

void JITPersistentCache::CleanupBySize(size_t target_size) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    if (stats_.total_disk_cache_size.load() <= target_size) {
        return;
    }
    
    // 按使用频率排序，优先保留常用内核
    std::vector<std::pair<std::string, const CacheMetadata*>> sorted_kernels;
    for (const auto& [key, metadata] : metadata_cache_) {
        sorted_kernels.emplace_back(key, &metadata);
    }
    
    std::sort(sorted_kernels.begin(), sorted_kernels.end(),
              [](const auto& a, const auto& b) {
                  // 优先保留使用次数多、最近使用的内核
                  if (a.second->usage_count != b.second->usage_count) {
                      return a.second->usage_count > b.second->usage_count;
                  }
                  return a.second->last_used_time > b.second->last_used_time;
              });
    
    // 从末尾开始移除，直到达到目标大小
    size_t current_size = stats_.total_disk_cache_size.load();
    for (auto it = sorted_kernels.rbegin(); it != sorted_kernels.rend(); ++it) {
        if (current_size <= target_size) {
            break;
        }
        
        RemoveKernel(it->first);
        current_size = stats_.total_disk_cache_size.load();
        stats_.cache_evictions.fetch_add(1);
    }
    
    LOG(INFO) << "Cache size cleanup completed. Current size: " 
              << stats_.total_disk_cache_size.load() << " bytes";
}

void JITPersistentCache::ValidateCacheIntegrity() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::vector<std::string> corrupted_keys;
    
    for (const auto& [key, metadata] : metadata_cache_) {
        // 检查PTX代码完整性
        auto ptx_it = ptx_cache_.find(key);
        if (ptx_it != ptx_cache_.end()) {
            if (metadata.ptx_code_hash != GeneratePTXHash(ptx_it->second)) {
                corrupted_keys.push_back(key);
                stats_.cache_corruptions.fetch_add(1);
            }
        }
    }
    
    // 移除损坏的缓存
    for (const auto& key : corrupted_keys) {
        RemoveKernel(key);
    }
    
    if (!corrupted_keys.empty()) {
        LOG(WARNING) << "Found and removed " << corrupted_keys.size() << " corrupted kernels";
    }
}

PersistentCacheStats JITPersistentCache::GetStats() const {
    return stats_;
}

void JITPersistentCache::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.Reset();
}

void JITPersistentCache::PrintStats() const {
    auto stats = GetStats();
    
    std::stringstream ss;
    ss << "\n=== JIT Persistent Cache Statistics ===\n";
    ss << "Total Cached Kernels: " << stats.total_cached_kernels.load() << "\n";
    ss << "Disk Cache Hits: " << stats.disk_cache_hits.load() << "\n";
    ss << "Disk Cache Misses: " << stats.disk_cache_misses.load() << "\n";
    ss << "Cache Hit Rate: " << std::fixed << std::setprecision(2)
       << (stats.disk_cache_hits.load() + stats.disk_cache_misses.load() > 0 ?
           (double)stats.disk_cache_hits.load() / 
           (stats.disk_cache_hits.load() + stats.disk_cache_misses.load()) * 100 : 0)
       << "%\n";
    ss << "Total Disk Cache Size: " << stats.total_disk_cache_size.load() << " bytes\n";
    ss << "Total Saved Compilation Time: " << stats.total_saved_compilation_time.load() << " ms\n";
    ss << "Cache Evictions: " << stats.cache_evictions.load() << "\n";
    ss << "Cache Corruptions: " << stats.cache_corruptions.load() << "\n";
    ss << "========================================\n";
    
    LOG(INFO) << ss.str();
}

// ==================== 私有方法实现 ====================

std::string JITPersistentCache::GenerateCacheFilePath(const std::string& cache_key) const {
    return cache_dir_ + "/kernels/" + cache_key + ".ptx";
}

std::string JITPersistentCache::GenerateMetadataFilePath(const std::string& cache_key) const {
    return cache_dir_ + "/metadata/" + cache_key + ".meta";
}

std::string JITPersistentCache::GenerateKernelCodeHash(const std::string& kernel_code) const {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, kernel_code.c_str(), kernel_code.length());
    SHA256_Final(hash, &sha256);
    
    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    return ss.str();
}

std::string JITPersistentCache::GeneratePTXHash(const std::string& ptx_code) const {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, ptx_code.c_str(), ptx_code.length());
    SHA256_Final(hash, &sha256);
    
    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    return ss.str();
}

std::string JITPersistentCache::GetCurrentEnvironmentSignature() const {
    std::stringstream ss;
    
    // CUDA版本
    int cuda_version;
    cudaDriverGetVersion(&cuda_version);
    ss << "CUDA_" << cuda_version << "_";
    
    // 计算能力
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    ss << "CC" << prop.major << prop.minor << "_";
    
    // 硬件信息
    ss << "SM" << prop.multiProcessorCount << "_";
    ss << "MEM" << (prop.totalGlobalMem / (1024 * 1024 * 1024)) << "GB";
    
    return ss.str();
}

bool JITPersistentCache::SaveMetadata(const std::string& cache_key, const CacheMetadata& metadata) {
    try {
        std::string filepath = GenerateMetadataFilePath(cache_key);
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
        
        // 序列化元数据
        // 这里简化实现，实际应该使用更robust的序列化
        file.write(metadata.cache_key.c_str(), metadata.cache_key.length());
        file.write("\0", 1);
        file.write(metadata.kernel_name.c_str(), metadata.kernel_name.length());
        file.write("\0", 1);
        
        // 保存编译选项
        for (const auto& option : metadata.compile_options) {
            file.write(option.c_str(), option.length());
            file.write("\0", 1);
        }
        file.write("\0\0", 2); // 选项列表结束标记
        
        // 保存其他字段...
        file.close();
        return true;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to save metadata: " << e.what();
        return false;
    }
}

bool JITPersistentCache::LoadMetadata(const std::string& cache_key, CacheMetadata& metadata) {
    try {
        std::string filepath = GenerateMetadataFilePath(cache_key);
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
        
        // 反序列化元数据
        // 这里简化实现，实际应该使用更robust的反序列化
        std::string line;
        std::getline(file, line, '\0');
        metadata.cache_key = line;
        
        std::getline(file, line, '\0');
        metadata.kernel_name = line;
        
        // 加载编译选项
        metadata.compile_options.clear();
        while (std::getline(file, line, '\0') && !line.empty()) {
            metadata.compile_options.push_back(line);
        }
        
        file.close();
        return true;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to load metadata: " << e.what();
        return false;
    }
}

bool JITPersistentCache::SavePTXCode(const std::string& cache_key, const std::string& ptx_code) {
    try {
        std::string filepath = GenerateCacheFilePath(cache_key);
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
        
        if (policy_.enable_compression) {
            auto compressed_data = CompressData(ptx_code);
            file.write(reinterpret_cast<const char*>(compressed_data.data()), compressed_data.size());
        } else {
            file.write(ptx_code.c_str(), ptx_code.length());
        }
        
        file.close();
        return true;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to save PTX code: " << e.what();
        return false;
    }
}

bool JITPersistentCache::LoadPTXCode(const std::string& cache_key, std::string& ptx_code) {
    try {
        std::string filepath = GenerateCacheFilePath(cache_key);
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
        
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> buffer(file_size);
        file.read(buffer.data(), file_size);
        file.close();
        
        if (policy_.enable_compression) {
            std::vector<uint8_t> compressed_data(buffer.begin(), buffer.end());
            if (!DecompressData(compressed_data, ptx_code)) {
                return false;
            }
        } else {
            ptx_code.assign(buffer.begin(), buffer.end());
        }
        
        return true;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to load PTX code: " << e.what();
        return false;
    }
}

void JITPersistentCache::UpdateUsageStatistics(const std::string& cache_key) {
    auto it = metadata_cache_.find(cache_key);
    if (it != metadata_cache_.end()) {
        it->second.last_used_time = std::chrono::system_clock::now();
        it->second.usage_count++;
    }
}

void JITPersistentCache::PerformCacheMaintenance() {
    auto now = std::chrono::system_clock::now();
    
    if (now - last_maintenance_time_ > policy_.cleanup_interval) {
        CleanupExpiredCache();
        ValidateCacheIntegrity();
        
        if (stats_.total_disk_cache_size.load() > policy_.max_disk_cache_size * policy_.eviction_threshold) {
            CleanupBySize(policy_.max_disk_cache_size * 0.7); // 清理到70%
        }
        
        last_maintenance_time_ = now;
        SaveStatistics();
    }
}

void JITPersistentCache::StartMaintenanceThread() {
    stop_maintenance_ = false;
    maintenance_thread_ = std::thread([this]() {
        while (!stop_maintenance_) {
            PerformCacheMaintenance();
            std::this_thread::sleep_for(std::chrono::minutes(5)); // 每5分钟检查一次
        }
    });
}

void JITPersistentCache::StopMaintenanceThread() {
    stop_maintenance_ = true;
    if (maintenance_thread_.joinable()) {
        maintenance_thread_.join();
    }
}

void JITPersistentCache::ScanExistingCache() {
    try {
        std::string kernels_dir = cache_dir_ + "/kernels";
        std::string metadata_dir = cache_dir_ + "/metadata";
        
        if (!std::filesystem::exists(kernels_dir) || !std::filesystem::exists(metadata_dir)) {
            return;
        }
        
        size_t scanned_count = 0;
        for (const auto& entry : std::filesystem::directory_iterator(metadata_dir)) {
            if (entry.path().extension() == ".meta") {
                std::string cache_key = entry.path().stem().string();
                
                // 检查对应的PTX文件是否存在
                std::string ptx_path = kernels_dir + "/" + cache_key + ".ptx";
                if (std::filesystem::exists(ptx_path)) {
                    CacheMetadata metadata;
                    if (LoadMetadata(cache_key, metadata)) {
                        metadata_cache_[cache_key] = metadata;
                        scanned_count++;
                    }
                }
            }
        }
        
        LOG(INFO) << "Scanned " << scanned_count << " existing cached kernels";
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to scan existing cache: " << e.what();
    }
}

void JITPersistentCache::SaveStatistics() {
    try {
        std::string stats_file = cache_dir_ + "/cache_stats.bin";
        std::ofstream file(stats_file, std::ios::binary);
        if (file.is_open()) {
            // 保存统计信息到文件
            file.close();
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to save statistics: " << e.what();
    }
}

// 压缩相关实现
std::vector<uint8_t> JITPersistentCache::CompressData(const std::string& data) const {
    std::vector<uint8_t> compressed;
    
    uLong compressed_size = compressBound(data.length());
    compressed.resize(compressed_size);
    
    if (compress2(compressed.data(), &compressed_size,
                  reinterpret_cast<const Bytef*>(data.data()), data.length(),
                  Z_BEST_SPEED) == Z_OK) {
        compressed.resize(compressed_size);
    } else {
        // 压缩失败，返回原始数据
        compressed.assign(data.begin(), data.end());
    }
    
    return compressed;
}

bool JITPersistentCache::DecompressData(const std::vector<uint8_t>& compressed_data, std::string& data) const {
    uLong decompressed_size = compressed_data.size() * 4; // 估计解压后大小
    std::vector<uint8_t> decompressed(decompressed_size);
    
    while (true) {
        uLong actual_size = decompressed_size;
        int result = uncompress(decompressed.data(), &actual_size,
                               compressed_data.data(), compressed_data.size());
        
        if (result == Z_OK) {
            decompressed.resize(actual_size);
            data.assign(decompressed.begin(), decompressed.end());
            return true;
        } else if (result == Z_BUF_ERROR) {
            // 缓冲区太小，增加大小重试
            decompressed_size *= 2;
            decompressed.resize(decompressed_size);
        } else {
            // 解压失败
            return false;
        }
    }
}

// 校验和实现
std::string JITPersistentCache::CalculateChecksum(const std::string& data) const {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, data.c_str(), data.length());
    SHA256_Final(hash, &sha256);
    
    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    return ss.str();
}

bool JITPersistentCache::VerifyChecksum(const std::string& data, const std::string& expected_checksum) const {
    std::string actual_checksum = CalculateChecksum(data);
    return actual_checksum == expected_checksum;
}

// ==================== GlobalPersistentCacheManager 实现 ====================

GlobalPersistentCacheManager& GlobalPersistentCacheManager::Instance() {
    static GlobalPersistentCacheManager instance;
    return instance;
}

bool GlobalPersistentCacheManager::Initialize(const std::string& cache_dir, const CachePolicy& policy) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        LOG(WARNING) << "Global Persistent Cache Manager already initialized";
        return true;
    }
    
    try {
        cache_dir_ = cache_dir;
        policy_ = policy;
        
        cache_manager_ = std::make_unique<JITPersistentCache>(cache_dir_, policy_);
        initialized_ = true;
        
        LOG(INFO) << "Global Persistent Cache Manager initialized successfully";
        return true;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to initialize Global Persistent Cache Manager: " << e.what();
        return false;
    }
}

void GlobalPersistentCacheManager::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (cache_manager_) {
        cache_manager_.reset();
    }
    
    initialized_ = false;
    LOG(INFO) << "Global Persistent Cache Manager cleaned up";
}

bool GlobalPersistentCacheManager::SaveKernel(const std::string& cache_key,
                                            const std::string& kernel_name,
                                            const std::string& kernel_code,
                                            const std::vector<std::string>& compile_options,
                                            const std::string& ptx_code,
                                            double compilation_time_ms) {
    if (!initialized_ || !cache_manager_) {
        return false;
    }
    
    return cache_manager_->SaveKernel(cache_key, kernel_name, kernel_code, 
                                    compile_options, ptx_code, compilation_time_ms);
}

bool GlobalPersistentCacheManager::LoadKernel(const std::string& cache_key,
                                            std::string& kernel_name,
                                            std::vector<std::string>& compile_options,
                                            std::string& ptx_code) {
    if (!initialized_ || !cache_manager_) {
        return false;
    }
    
    return cache_manager_->LoadKernel(cache_key, kernel_name, compile_options, ptx_code);
}

bool GlobalPersistentCacheManager::IsKernelCached(const std::string& cache_key) const {
    if (!initialized_ || !cache_manager_) {
        return false;
    }
    
    return cache_manager_->IsKernelCached(cache_key);
}

PersistentCacheStats GlobalPersistentCacheManager::GetStats() const {
    if (!initialized_ || !cache_manager_) {
        return PersistentCacheStats{};
    }
    
    return cache_manager_->GetStats();
}

CachePolicy GlobalPersistentCacheManager::GetCachePolicy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return policy_;
}

void GlobalPersistentCacheManager::SetCachePolicy(const CachePolicy& policy) {
    std::lock_guard<std::mutex> lock(mutex_);
    policy_ = policy;
    
    if (cache_manager_) {
        cache_manager_->SetCachePolicy(policy);
    }
}

void GlobalPersistentCacheManager::CleanupExpiredCache() {
    if (initialized_ && cache_manager_) {
        cache_manager_->CleanupExpiredCache();
    }
}

void GlobalPersistentCacheManager::ValidateCacheIntegrity() {
    if (initialized_ && cache_manager_) {
        cache_manager_->ValidateCacheIntegrity();
    }
}

} // namespace cu_op_mem 