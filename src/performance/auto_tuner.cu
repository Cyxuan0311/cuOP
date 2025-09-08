#include "performance/auto_tuner.hpp"
#include <glog/logging.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <json/json.h>

namespace cu_op_mem {

AutoTuner& AutoTuner::Instance() {
    static AutoTuner instance;
    return instance;
}

AutoTuner::AutoTuner() 
    : tuning_enabled_(false), real_time_monitoring_(false), monitoring_interval_ms_(1000) {
    // 初始化默认调优策略
    tuning_strategy_ = std::make_unique<RandomSearchStrategy>();
}

AutoTuner::~AutoTuner() {
    StopTuning();
}

void AutoTuner::StartTuning() {
    if (tuning_enabled_) {
        LOG(WARNING) << "Auto-tuning is already enabled";
        return;
    }
    
    tuning_enabled_ = true;
    
    // 启动后台调优线程
    if (real_time_monitoring_) {
        tuning_thread_ = std::thread(&AutoTuner::BackgroundTuning, this);
    }
    
    LOG(INFO) << "Auto-tuning started";
}

void AutoTuner::StopTuning() {
    if (!tuning_enabled_) {
        return;
    }
    
    tuning_enabled_ = false;
    
    // 停止后台调优线程
    if (tuning_thread_.joinable()) {
        tuning_thread_.join();
    }
    
    LOG(INFO) << "Auto-tuning stopped";
}

void AutoTuner::Reset() {
    std::lock_guard<std::mutex> config_lock(config_mutex_);
    std::lock_guard<std::mutex> results_lock(results_mutex_);
    
    parameters_.clear();
    tuning_history_.clear();
    performance_cache_.clear();
    
    LOG(INFO) << "Auto-tuner reset";
}

void AutoTuner::SetTuningConfig(const TuningConfig& config) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    tuning_config_ = config;
}

TuningConfig AutoTuner::GetTuningConfig() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return tuning_config_;
}

void AutoTuner::AddParameter(const ParameterDefinition& param) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    parameters_.push_back(param);
}

void AutoTuner::RemoveParameter(const std::string& param_name) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    parameters_.erase(
        std::remove_if(parameters_.begin(), parameters_.end(),
            [&param_name](const ParameterDefinition& param) {
                return param.name == param_name;
            }),
        parameters_.end());
}

std::vector<ParameterDefinition> AutoTuner::GetParameters() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return parameters_;
}

TuningResult AutoTuner::Tune(const std::string& operator_name, BenchmarkFunction benchmark_func) {
    std::lock_guard<std::mutex> config_lock(config_mutex_);
    std::lock_guard<std::mutex> results_lock(results_mutex_);
    
    TuningResult result;
    result.tuning_start_time = std::chrono::high_resolution_clock::now();
    result.best_performance = 0.0;
    result.iterations_used = 0;
    result.converged = false;
    
    LOG(INFO) << "Starting auto-tuning for operator: " << operator_name;
    
    // 生成参数组合
    std::vector<std::unordered_map<std::string, std::string>> parameter_sets;
    
    if (tuning_strategy_) {
        parameter_sets = tuning_strategy_->GenerateParameterSets(tuning_config_, tuning_config_.max_iterations);
    } else {
        LOG(ERROR) << "No tuning strategy set";
        return result;
    }
    
    // 限制参数组合数量
    if (parameter_sets.size() > tuning_config_.max_iterations) {
        parameter_sets.resize(tuning_config_.max_iterations);
    }
    
    // 测试每个参数组合
    for (size_t i = 0; i < parameter_sets.size(); ++i) {
        const auto& params = parameter_sets[i];
        
        try {
            double performance = benchmark_func(params);
            result.performance_history.push_back(performance);
            result.parameter_history.push_back(params);
            
            if (performance > result.best_performance) {
                result.best_performance = performance;
                result.best_parameters = params;
            }
            
            LOG(INFO) << "Iteration " << i + 1 << "/" << parameter_sets.size() 
                     << " - Performance: " << performance 
                     << " - Best: " << result.best_performance;
            
            // 更新调优策略
            if (tuning_strategy_) {
                tuning_strategy_->UpdateStrategy(result.performance_history, result.parameter_history);
            }
            
            // 检查早停
            if (tuning_config_.enable_early_stopping && i >= tuning_config_.patience) {
                if (CheckConvergence(result.performance_history)) {
                    result.converged = true;
                    break;
                }
            }
            
        } catch (const std::exception& e) {
            LOG(WARNING) << "Benchmark failed for iteration " << i + 1 << ": " << e.what();
            result.performance_history.push_back(0.0);
            result.parameter_history.push_back(params);
        }
        
        result.iterations_used = i + 1;
    }
    
    // 计算收敛率
    result.convergence_rate = CalculateConvergenceRate(result.performance_history);
    
    // 生成优化摘要
    result.optimization_summary = GenerateOptimizationSummary(result);
    
    // 生成建议
    result.recommendations = GenerateRecommendations(result);
    
    result.tuning_end_time = std::chrono::high_resolution_clock::now();
    
    // 保存到历史
    tuning_history_.push_back(result);
    
    LOG(INFO) << "Auto-tuning completed for " << operator_name;
    return result;
}

TuningResult AutoTuner::TuneAsync(const std::string& operator_name, BenchmarkFunction benchmark_func) {
    // 异步调优实现
    std::future<TuningResult> future_result = std::async(std::launch::async, 
        [this, operator_name, benchmark_func]() {
            return Tune(operator_name, benchmark_func);
        });
    
    return future_result.get();
}

void AutoTuner::SetTuningStrategy(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(strategy_mutex_);
    
    if (strategy_name == "GridSearch") {
        tuning_strategy_ = std::make_unique<GridSearchStrategy>();
    } else if (strategy_name == "RandomSearch") {
        tuning_strategy_ = std::make_unique<RandomSearchStrategy>();
    } else if (strategy_name == "BayesianOptimization") {
        tuning_strategy_ = std::make_unique<BayesianOptimizationStrategy>();
    } else if (strategy_name == "GeneticAlgorithm") {
        tuning_strategy_ = std::make_unique<GeneticAlgorithmStrategy>();
    } else {
        LOG(WARNING) << "Unknown tuning strategy: " << strategy_name;
        return;
    }
    
    LOG(INFO) << "Tuning strategy set to: " << strategy_name;
}

std::vector<std::string> AutoTuner::GetAvailableStrategies() const {
    return {"GridSearch", "RandomSearch", "BayesianOptimization", "GeneticAlgorithm"};
}

void AutoTuner::SaveTuningHistory(const std::string& filename) {
    std::lock_guard<std::mutex> lock(results_mutex_);
    
    Json::Value root(Json::arrayValue);
    
    for (const auto& result : tuning_history_) {
        Json::Value result_json;
        result_json["operator_name"] = tuning_config_.operator_name;
        result_json["best_performance"] = result.best_performance;
        result_json["iterations_used"] = result.iterations_used;
        result_json["converged"] = result.converged;
        result_json["convergence_rate"] = result.convergence_rate;
        
        // 最佳参数
        Json::Value best_params;
        for (const auto& pair : result.best_parameters) {
            best_params[pair.first] = pair.second;
        }
        result_json["best_parameters"] = best_params;
        
        // 性能历史
        Json::Value performance_history(Json::arrayValue);
        for (double perf : result.performance_history) {
            performance_history.append(perf);
        }
        result_json["performance_history"] = performance_history;
        
        root.append(result_json);
    }
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open file for tuning history save: " << filename;
        return;
    }
    
    file << root;
    file.close();
    LOG(INFO) << "Tuning history saved to: " << filename;
}

void AutoTuner::LoadTuningHistory(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open file for tuning history load: " << filename;
        return;
    }
    
    Json::Value root;
    file >> root;
    
    std::lock_guard<std::mutex> lock(results_mutex_);
    tuning_history_.clear();
    
    for (const auto& result_json : root) {
        TuningResult result;
        result.best_performance = result_json["best_performance"].asDouble();
        result.iterations_used = result_json["iterations_used"].asUInt();
        result.converged = result_json["converged"].asBool();
        result.convergence_rate = result_json["convergence_rate"].asDouble();
        
        // 加载最佳参数
        const Json::Value& best_params = result_json["best_parameters"];
        for (const auto& key : best_params.getMemberNames()) {
            result.best_parameters[key] = best_params[key].asString();
        }
        
        // 加载性能历史
        const Json::Value& performance_history = result_json["performance_history"];
        for (const auto& perf : performance_history) {
            result.performance_history.push_back(perf.asDouble());
        }
        
        tuning_history_.push_back(result);
    }
    
    LOG(INFO) << "Tuning history loaded from: " << filename;
}

std::vector<TuningResult> AutoTuner::GetTuningHistory() const {
    std::lock_guard<std::mutex> lock(results_mutex_);
    return tuning_history_;
}

double AutoTuner::PredictPerformance(const std::unordered_map<std::string, std::string>& parameters) {
    std::lock_guard<std::mutex> cache_lock(cache_mutex_);
    
    // 简化的性能预测：基于历史数据
    std::string param_key;
    for (const auto& pair : parameters) {
        param_key += pair.first + "=" + pair.second + ";";
    }
    
    auto it = performance_cache_.find(param_key);
    if (it != performance_cache_.end()) {
        return it->second;
    }
    
    // 基于历史数据预测
    if (!tuning_history_.empty()) {
        const auto& last_result = tuning_history_.back();
        return last_result.best_performance * 0.9; // 保守估计
    }
    
    return 0.0;
}

std::vector<std::string> AutoTuner::GetPerformanceInsights() {
    std::vector<std::string> insights;
    
    if (tuning_history_.empty()) {
        insights.push_back("No tuning history available");
        return insights;
    }
    
    const auto& latest_result = tuning_history_.back();
    
    if (latest_result.converged) {
        insights.push_back("Tuning converged successfully");
    } else {
        insights.push_back("Tuning did not converge");
    }
    
    if (latest_result.convergence_rate > 0.8) {
        insights.push_back("High convergence rate achieved");
    } else if (latest_result.convergence_rate < 0.3) {
        insights.push_back("Low convergence rate - consider different strategy");
    }
    
    if (latest_result.performance_history.size() > 10) {
        double improvement = (latest_result.best_performance - latest_result.performance_history[0]) / 
                           latest_result.performance_history[0] * 100.0;
        insights.push_back("Performance improved by " + std::to_string(improvement) + "%");
    }
    
    return insights;
}

std::string AutoTuner::GenerateTuningReport(const TuningResult& result) {
    std::ostringstream report;
    report << "=== cuOP Auto-Tuning Report ===\n";
    report << "Operator: " << tuning_config_.operator_name << "\n";
    report << "Strategy: " << (tuning_strategy_ ? tuning_strategy_->GetStrategyName() : "Unknown") << "\n";
    report << "Iterations Used: " << result.iterations_used << "\n";
    report << "Best Performance: " << std::fixed << std::setprecision(2) << result.best_performance << "\n";
    report << "Converged: " << (result.converged ? "Yes" : "No") << "\n";
    report << "Convergence Rate: " << std::fixed << std::setprecision(2) << result.convergence_rate << "\n";
    
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        result.tuning_end_time - result.tuning_start_time);
    report << "Tuning Duration: " << duration.count() << " seconds\n\n";
    
    // 最佳参数
    report << "=== Best Parameters ===\n";
    for (const auto& pair : result.best_parameters) {
        report << pair.first << ": " << pair.second << "\n";
    }
    
    // 性能历史
    report << "\n=== Performance History ===\n";
    for (size_t i = 0; i < std::min(size_t(10), result.performance_history.size()); ++i) {
        report << "Iteration " << i + 1 << ": " << std::fixed << std::setprecision(2) 
               << result.performance_history[i] << "\n";
    }
    
    // 建议
    report << "\n=== Recommendations ===\n";
    for (size_t i = 0; i < result.recommendations.size(); ++i) {
        report << i + 1 << ". " << result.recommendations[i] << "\n";
    }
    
    return report.str();
}

void AutoTuner::ExportResults(const TuningResult& result, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open file for results export: " << filename;
        return;
    }
    
    file << GenerateTuningReport(result);
    file.close();
    LOG(INFO) << "Tuning results exported to: " << filename;
}

void AutoTuner::EnableRealTimeMonitoring(bool enable) {
    real_time_monitoring_ = enable;
    if (enable && tuning_enabled_) {
        tuning_thread_ = std::thread(&AutoTuner::BackgroundTuning, this);
    }
}

void AutoTuner::SetMonitoringInterval(int interval_ms) {
    monitoring_interval_ms_ = interval_ms;
}

void AutoTuner::RegisterCallback(std::function<void(const TuningResult&)> callback) {
    callbacks_.push_back(callback);
}

void AutoTuner::BackgroundTuning() {
    while (tuning_enabled_ && real_time_monitoring_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(monitoring_interval_ms_));
        
        if (tuning_enabled_) {
            ProcessTuningResults();
            UpdateTuningStrategy();
        }
    }
}

void AutoTuner::ProcessTuningResults() {
    // 处理调优结果
    VLOG(1) << "Processing tuning results";
}

void AutoTuner::UpdateTuningStrategy() {
    // 更新调优策略
    VLOG(1) << "Updating tuning strategy";
}

std::string AutoTuner::GenerateOptimizationSummary(const TuningResult& result) {
    std::ostringstream summary;
    summary << "Auto-tuning completed for " << tuning_config_.operator_name << "\n";
    summary << "Best performance: " << result.best_performance << "\n";
    summary << "Iterations used: " << result.iterations_used << "\n";
    summary << "Converged: " << (result.converged ? "Yes" : "No") << "\n";
    summary << "Convergence rate: " << result.convergence_rate << "\n";
    
    if (!result.best_parameters.empty()) {
        summary << "Best parameters:\n";
        for (const auto& pair : result.best_parameters) {
            summary << "  " << pair.first << ": " << pair.second << "\n";
        }
    }
    
    return summary.str();
}

std::vector<std::string> AutoTuner::GenerateRecommendations(const TuningResult& result) {
    std::vector<std::string> recommendations;
    
    if (!result.converged) {
        recommendations.push_back("Consider increasing max_iterations or adjusting convergence_threshold");
    }
    
    if (result.convergence_rate < 0.5) {
        recommendations.push_back("Consider using a different tuning strategy");
    }
    
    if (result.performance_history.size() > 5) {
        double improvement = (result.best_performance - result.performance_history[0]) / 
                           result.performance_history[0] * 100.0;
        if (improvement < 10.0) {
            recommendations.push_back("Limited performance improvement - consider parameter range adjustment");
        }
    }
    
    return recommendations;
}

bool AutoTuner::CheckConvergence(const std::vector<double>& performance_history) {
    if (performance_history.size() < tuning_config_.patience) {
        return false;
    }
    
    // 检查最近几次迭代的性能变化
    size_t start_idx = performance_history.size() - tuning_config_.patience;
    double min_perf = *std::min_element(performance_history.begin() + start_idx, performance_history.end());
    double max_perf = *std::max_element(performance_history.begin() + start_idx, performance_history.end());
    
    double improvement = (max_perf - min_perf) / max_perf;
    return improvement < tuning_config_.convergence_threshold;
}

double AutoTuner::CalculateConvergenceRate(const std::vector<double>& performance_history) {
    if (performance_history.size() < 2) {
        return 0.0;
    }
    
    // 计算性能变化的稳定性
    std::vector<double> differences;
    for (size_t i = 1; i < performance_history.size(); ++i) {
        differences.push_back(std::abs(performance_history[i] - performance_history[i-1]));
    }
    
    double avg_difference = std::accumulate(differences.begin(), differences.end(), 0.0) / differences.size();
    double max_difference = *std::max_element(differences.begin(), differences.end());
    
    return max_difference > 0 ? (1.0 - avg_difference / max_difference) : 1.0;
}

// 调优策略实现

std::vector<std::unordered_map<std::string, std::string>> GridSearchStrategy::GenerateParameterSets(
    const TuningConfig& config, size_t count) {
    std::vector<std::unordered_map<std::string, std::string>> combinations;
    GenerateCombinations(config.parameters, {}, combinations);
    
    // 限制组合数量
    if (combinations.size() > count) {
        combinations.resize(count);
    }
    
    return combinations;
}

void GridSearchStrategy::GenerateCombinations(const std::vector<ParameterDefinition>& parameters,
                                             std::unordered_map<std::string, std::string> current_params,
                                             std::vector<std::unordered_map<std::string, std::string>>& combinations) {
    if (parameters.empty()) {
        combinations.push_back(current_params);
        return;
    }
    
    const auto& param = parameters[0];
    std::vector<ParameterDefinition> remaining_params(parameters.begin() + 1, parameters.end());
    
    for (const auto& value : param.values) {
        current_params[param.name] = value;
        GenerateCombinations(remaining_params, current_params, combinations);
    }
}

void GridSearchStrategy::UpdateStrategy(const std::vector<double>& performance_history,
                                       const std::vector<std::unordered_map<std::string, std::string>>& parameter_history) {
    // 网格搜索不需要更新策略
}

RandomSearchStrategy::RandomSearchStrategy() : rng_(std::random_device{}()) {}

std::vector<std::unordered_map<std::string, std::string>> RandomSearchStrategy::GenerateParameterSets(
    const TuningConfig& config, size_t count) {
    std::vector<std::unordered_map<std::string, std::string>> parameter_sets;
    
    for (size_t i = 0; i < count; ++i) {
        std::unordered_map<std::string, std::string> params;
        
        for (const auto& param : config.parameters) {
            if (param.is_tunable) {
                params[param.name] = GenerateRandomValue(param);
            }
        }
        
        parameter_sets.push_back(params);
    }
    
    return parameter_sets;
}

std::string RandomSearchStrategy::GenerateRandomValue(const ParameterDefinition& param) {
    if (param.values.empty()) {
        return param.default_value;
    }
    
    std::uniform_int_distribution<size_t> dist(0, param.values.size() - 1);
    return param.values[dist(rng_)];
}

void RandomSearchStrategy::UpdateStrategy(const std::vector<double>& performance_history,
                                         const std::vector<std::unordered_map<std::string, std::string>>& parameter_history) {
    // 随机搜索不需要更新策略
}

BayesianOptimizationStrategy::BayesianOptimizationStrategy() : rng_(std::random_device{}()) {}

std::vector<std::unordered_map<std::string, std::string>> BayesianOptimizationStrategy::GenerateParameterSets(
    const TuningConfig& config, size_t count) {
    std::vector<std::unordered_map<std::string, std::string>> parameter_sets;
    
    for (size_t i = 0; i < count; ++i) {
        std::unordered_map<std::string, std::string> params;
        
        for (const auto& param : config.parameters) {
            if (param.is_tunable) {
                params[param.name] = GenerateRandomValue(param);
            }
        }
        
        parameter_sets.push_back(params);
    }
    
    return parameter_sets;
}

std::string BayesianOptimizationStrategy::GenerateRandomValue(const ParameterDefinition& param) {
    if (param.values.empty()) {
        return param.default_value;
    }
    
    std::uniform_int_distribution<size_t> dist(0, param.values.size() - 1);
    return param.values[dist(rng_)];
}

void BayesianOptimizationStrategy::UpdateStrategy(const std::vector<double>& performance_history,
                                                 const std::vector<std::unordered_map<std::string, std::string>>& parameter_history) {
    performance_history_ = performance_history;
    parameter_history_ = parameter_history;
}

double BayesianOptimizationStrategy::CalculateAcquisitionFunction(const std::unordered_map<std::string, std::string>& params) {
    return CalculateExpectedImprovement(params);
}

double BayesianOptimizationStrategy::CalculateExpectedImprovement(const std::unordered_map<std::string, std::string>& params) {
    if (performance_history_.empty()) {
        return 1.0;
    }
    
    double best_performance = *std::max_element(performance_history_.begin(), performance_history_.end());
    // 简化的期望改进计算
    return best_performance * 0.1;
}

GeneticAlgorithmStrategy::GeneticAlgorithmStrategy() : rng_(std::random_device{}()) {}

std::vector<std::unordered_map<std::string, std::string>> GeneticAlgorithmStrategy::GenerateParameterSets(
    const TuningConfig& config, size_t count) {
    if (population_.empty()) {
        InitializePopulation(config, count);
    }
    
    return population_;
}

void GeneticAlgorithmStrategy::InitializePopulation(const TuningConfig& config, size_t population_size) {
    population_.clear();
    fitness_scores_.clear();
    
    for (size_t i = 0; i < population_size; ++i) {
        std::unordered_map<std::string, std::string> individual;
        
        for (const auto& param : config.parameters) {
            if (param.is_tunable) {
                std::uniform_int_distribution<size_t> dist(0, param.values.size() - 1);
                individual[param.name] = param.values[dist(rng_)];
            }
        }
        
        population_.push_back(individual);
        fitness_scores_.push_back(0.0);
    }
}

void GeneticAlgorithmStrategy::UpdateStrategy(const std::vector<double>& performance_history,
                                             const std::vector<std::unordered_map<std::string, std::string>>& parameter_history) {
    if (performance_history.empty() || parameter_history.empty()) {
        return;
    }
    
    // 更新适应度分数
    for (size_t i = 0; i < std::min(population_.size(), performance_history.size()); ++i) {
        fitness_scores_[i] = performance_history[i];
    }
    
    // 执行遗传算法操作
    Selection();
    Crossover();
    Mutation(TuningConfig{}); // 简化实现
}

void GeneticAlgorithmStrategy::Selection() {
    // 简化的选择操作
    std::vector<std::unordered_map<std::string, std::string>> new_population;
    
    for (size_t i = 0; i < population_.size(); ++i) {
        // 选择适应度最高的个体
        auto max_it = std::max_element(fitness_scores_.begin(), fitness_scores_.end());
        size_t max_idx = std::distance(fitness_scores_.begin(), max_it);
        new_population.push_back(population_[max_idx]);
    }
    
    population_ = new_population;
}

void GeneticAlgorithmStrategy::Crossover() {
    // 简化的交叉操作
    for (size_t i = 0; i < population_.size() - 1; i += 2) {
        auto child = CrossoverParameters(population_[i], population_[i + 1]);
        population_[i] = child;
    }
}

void GeneticAlgorithmStrategy::Mutation(const TuningConfig& config) {
    // 简化的变异操作
    for (auto& individual : population_) {
        for (const auto& param : config.parameters) {
            if (param.is_tunable && std::uniform_real_distribution<double>(0, 1)(rng_) < 0.1) {
                MutateParameter(individual, param);
            }
        }
    }
}

std::unordered_map<std::string, std::string> GeneticAlgorithmStrategy::CrossoverParameters(
    const std::unordered_map<std::string, std::string>& parent1,
    const std::unordered_map<std::string, std::string>& parent2) {
    std::unordered_map<std::string, std::string> child;
    
    for (const auto& pair : parent1) {
        if (std::uniform_real_distribution<double>(0, 1)(rng_) < 0.5) {
            child[pair.first] = pair.second;
        } else {
            auto it = parent2.find(pair.first);
            if (it != parent2.end()) {
                child[pair.first] = it->second;
            }
        }
    }
    
    return child;
}

void GeneticAlgorithmStrategy::MutateParameter(std::unordered_map<std::string, std::string>& params, 
                                              const ParameterDefinition& param_def) {
    if (!param_def.values.empty()) {
        std::uniform_int_distribution<size_t> dist(0, param_def.values.size() - 1);
        params[param_def.name] = param_def.values[dist(rng_)];
    }
}

} // namespace cu_op_mem
