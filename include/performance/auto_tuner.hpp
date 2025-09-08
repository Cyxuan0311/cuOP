#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <atomic>
#include <chrono>
#include <functional>
#include <random>
#include <algorithm>
#include <thread>
#include <future>

namespace cu_op_mem {

// 参数类型
enum class ParameterType {
    INTEGER,        // 整数参数
    FLOAT,          // 浮点数参数
    STRING,         // 字符串参数
    BOOLEAN,        // 布尔参数
    ENUM            // 枚举参数
};

// 参数定义
struct ParameterDefinition {
    std::string name;                   // 参数名
    ParameterType type;                 // 参数类型
    std::string description;            // 参数描述
    std::vector<std::string> values;    // 参数值列表
    std::string default_value;          // 默认值
    std::string min_value;              // 最小值
    std::string max_value;              // 最大值
    double weight;                      // 参数权重
    bool is_tunable;                    // 是否可调优
};

// 调优配置
struct TuningConfig {
    std::string operator_name;          // 算子名称
    std::vector<ParameterDefinition> parameters; // 参数定义
    std::string optimization_target;    // 优化目标 (throughput, latency, memory, accuracy)
    size_t max_iterations;              // 最大迭代次数
    double convergence_threshold;       // 收敛阈值
    double improvement_threshold;       // 改进阈值
    bool enable_parallel_tuning;        // 启用并行调优
    size_t parallel_workers;            // 并行工作线程数
    bool enable_early_stopping;         // 启用早停
    size_t patience;                    // 早停耐心值
    std::string tuning_strategy;        // 调优策略 (grid, random, bayesian, genetic)
    std::string result_save_path;       // 结果保存路径
};

// 调优结果
struct TuningResult {
    std::unordered_map<std::string, std::string> best_parameters; // 最佳参数
    double best_performance;            // 最佳性能
    size_t iterations_used;             // 使用的迭代次数
    std::vector<double> performance_history; // 性能历史
    std::vector<std::unordered_map<std::string, std::string>> parameter_history; // 参数历史
    std::string optimization_summary;   // 优化摘要
    bool converged;                     // 是否收敛
    double convergence_rate;            // 收敛率
    std::chrono::high_resolution_clock::time_point tuning_start_time;
    std::chrono::high_resolution_clock::time_point tuning_end_time;
    std::vector<std::string> recommendations; // 优化建议
};

// 性能基准函数类型
using BenchmarkFunction = std::function<double(const std::unordered_map<std::string, std::string>&)>;

// 调优策略接口
class ITuningStrategy {
public:
    virtual ~ITuningStrategy() = default;
    virtual std::vector<std::unordered_map<std::string, std::string>> GenerateParameterSets(
        const TuningConfig& config, size_t count) = 0;
    virtual void UpdateStrategy(const std::vector<double>& performance_history,
                               const std::vector<std::unordered_map<std::string, std::string>>& parameter_history) = 0;
    virtual std::string GetStrategyName() const = 0;
};

// 网格搜索策略
class GridSearchStrategy : public ITuningStrategy {
public:
    std::vector<std::unordered_map<std::string, std::string>> GenerateParameterSets(
        const TuningConfig& config, size_t count) override;
    void UpdateStrategy(const std::vector<double>& performance_history,
                       const std::vector<std::unordered_map<std::string, std::string>>& parameter_history) override;
    std::string GetStrategyName() const override { return "GridSearch"; }
    
private:
    void GenerateCombinations(const std::vector<ParameterDefinition>& parameters,
                             std::unordered_map<std::string, std::string> current_params,
                             std::vector<std::unordered_map<std::string, std::string>>& combinations);
};

// 随机搜索策略
class RandomSearchStrategy : public ITuningStrategy {
public:
    RandomSearchStrategy();
    std::vector<std::unordered_map<std::string, std::string>> GenerateParameterSets(
        const TuningConfig& config, size_t count) override;
    void UpdateStrategy(const std::vector<double>& performance_history,
                       const std::vector<std::unordered_map<std::string, std::string>>& parameter_history) override;
    std::string GetStrategyName() const override { return "RandomSearch"; }
    
private:
    std::mt19937 rng_;
    std::string GenerateRandomValue(const ParameterDefinition& param);
};

// 贝叶斯优化策略
class BayesianOptimizationStrategy : public ITuningStrategy {
public:
    BayesianOptimizationStrategy();
    std::vector<std::unordered_map<std::string, std::string>> GenerateParameterSets(
        const TuningConfig& config, size_t count) override;
    void UpdateStrategy(const std::vector<double>& performance_history,
                       const std::vector<std::unordered_map<std::string, std::string>>& parameter_history) override;
    std::string GetStrategyName() const override { return "BayesianOptimization"; }
    
private:
    std::mt19937 rng_;
    std::vector<double> performance_history_;
    std::vector<std::unordered_map<std::string, std::string>> parameter_history_;
    
    std::string GenerateRandomValue(const ParameterDefinition& param);
    double CalculateAcquisitionFunction(const std::unordered_map<std::string, std::string>& params);
    double CalculateExpectedImprovement(const std::unordered_map<std::string, std::string>& params);
};

// 遗传算法策略
class GeneticAlgorithmStrategy : public ITuningStrategy {
public:
    GeneticAlgorithmStrategy();
    std::vector<std::unordered_map<std::string, std::string>> GenerateParameterSets(
        const TuningConfig& config, size_t count) override;
    void UpdateStrategy(const std::vector<double>& performance_history,
                       const std::vector<std::unordered_map<std::string, std::string>>& parameter_history) override;
    std::string GetStrategyName() const override { return "GeneticAlgorithm"; }
    
private:
    std::mt19937 rng_;
    std::vector<std::unordered_map<std::string, std::string>> population_;
    std::vector<double> fitness_scores_;
    
    void InitializePopulation(const TuningConfig& config, size_t population_size);
    void Selection();
    void Crossover();
    void Mutation(const TuningConfig& config);
    std::unordered_map<std::string, std::string> CrossoverParameters(
        const std::unordered_map<std::string, std::string>& parent1,
        const std::unordered_map<std::string, std::string>& parent2);
    void MutateParameter(std::unordered_map<std::string, std::string>& params, 
                        const ParameterDefinition& param_def);
};

// 自动调优器主类
class AutoTuner {
public:
    static AutoTuner& Instance();
    
    // 基本控制
    void StartTuning();
    void StopTuning();
    void Reset();
    bool IsTuning() const { return tuning_enabled_; }
    
    // 调优配置
    void SetTuningConfig(const TuningConfig& config);
    TuningConfig GetTuningConfig() const;
    
    // 参数定义
    void AddParameter(const ParameterDefinition& param);
    void RemoveParameter(const std::string& param_name);
    std::vector<ParameterDefinition> GetParameters() const;
    
    // 调优执行
    TuningResult Tune(const std::string& operator_name, BenchmarkFunction benchmark_func);
    TuningResult TuneAsync(const std::string& operator_name, BenchmarkFunction benchmark_func);
    
    // 调优策略
    void SetTuningStrategy(const std::string& strategy_name);
    std::vector<std::string> GetAvailableStrategies() const;
    
    // 历史数据管理
    void SaveTuningHistory(const std::string& filename);
    void LoadTuningHistory(const std::string& filename);
    std::vector<TuningResult> GetTuningHistory() const;
    
    // 性能预测
    double PredictPerformance(const std::unordered_map<std::string, std::string>& parameters);
    std::vector<std::string> GetPerformanceInsights();
    
    // 结果分析
    std::string GenerateTuningReport(const TuningResult& result);
    void ExportResults(const TuningResult& result, const std::string& filename);
    
    // 实时监控
    void EnableRealTimeMonitoring(bool enable);
    void SetMonitoringInterval(int interval_ms);
    void RegisterCallback(std::function<void(const TuningResult&)> callback);
    
private:
    AutoTuner();
    ~AutoTuner();
    
    // 禁用拷贝构造和赋值
    AutoTuner(const AutoTuner&) = delete;
    AutoTuner& operator=(const AutoTuner&) = delete;
    
    // 内部方法
    void BackgroundTuning();
    void ProcessTuningResults();
    void UpdateTuningStrategy();
    std::string GenerateOptimizationSummary(const TuningResult& result);
    std::vector<std::string> GenerateRecommendations(const TuningResult& result);
    bool CheckConvergence(const std::vector<double>& performance_history);
    double CalculateConvergenceRate(const std::vector<double>& performance_history);
    
    // 成员变量
    std::atomic<bool> tuning_enabled_;
    std::atomic<bool> real_time_monitoring_;
    mutable std::mutex config_mutex_;
    mutable std::mutex results_mutex_;
    mutable std::mutex strategy_mutex_;
    
    TuningConfig tuning_config_;
    std::vector<ParameterDefinition> parameters_;
    std::unique_ptr<ITuningStrategy> tuning_strategy_;
    std::vector<TuningResult> tuning_history_;
    
    // 实时监控
    std::thread tuning_thread_;
    std::atomic<int> monitoring_interval_ms_;
    std::vector<std::function<void(const TuningResult&)>> callbacks_;
    
    // 性能预测
    std::unordered_map<std::string, double> performance_cache_;
    std::mutex cache_mutex_;
};

// 自动调优宏
#define CUOP_AUTO_TUNE(op_name, benchmark_func) \
    AutoTuner::Instance().Tune(op_name, benchmark_func)

#define CUOP_AUTO_TUNE_ASYNC(op_name, benchmark_func) \
    AutoTuner::Instance().TuneAsync(op_name, benchmark_func)

#define CUOP_ADD_PARAMETER(param_def) \
    AutoTuner::Instance().AddParameter(param_def)

#define CUOP_SET_TUNING_STRATEGY(strategy_name) \
    AutoTuner::Instance().SetTuningStrategy(strategy_name)

// 参数定义宏
#define CUOP_PARAM_INT(name, values, default_val, weight) \
    ParameterDefinition{name, ParameterType::INTEGER, "", values, default_val, "", "", weight, true}

#define CUOP_PARAM_FLOAT(name, values, default_val, weight) \
    ParameterDefinition{name, ParameterType::FLOAT, "", values, default_val, "", "", weight, true}

#define CUOP_PARAM_STRING(name, values, default_val, weight) \
    ParameterDefinition{name, ParameterType::STRING, "", values, default_val, "", "", weight, true}

#define CUOP_PARAM_BOOL(name, values, default_val, weight) \
    ParameterDefinition{name, ParameterType::BOOLEAN, "", values, default_val, "", "", weight, true}

#define CUOP_PARAM_ENUM(name, values, default_val, weight) \
    ParameterDefinition{name, ParameterType::ENUM, "", values, default_val, "", "", weight, true}

} // namespace cu_op_mem
