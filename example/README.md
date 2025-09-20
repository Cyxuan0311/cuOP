# cuOP 示例程序

本目录包含了cuOP库的各种使用示例，展示了如何在实际项目中使用cuOP的C++ API。

## 目录结构

```
example/
├── CMakeLists.txt              # 主CMake文件
├── README.md                   # 本文件
├── basic_operations/           # 基础操作示例
│   ├── CMakeLists.txt
│   ├── basic_gemm.cpp         # GEMM矩阵乘法示例
│   ├── basic_gemv.cpp         # GEMV矩阵向量乘法示例
│   ├── basic_dot.cpp          # DOT向量点积示例
│   └── basic_axpy.cpp         # AXPY向量加法示例
├── deep_learning/              # 深度学习示例
│   ├── CMakeLists.txt
│   ├── relu_demo.cpp          # ReLU激活函数示例
│   ├── softmax_demo.cpp       # Softmax归一化示例
│   └── conv2d_demo.cpp        # Conv2D卷积示例
├── model_inference/            # 模型推理示例
│   ├── CMakeLists.txt
│   └── simple_cnn_inference.cpp # 简单CNN推理演示
├── model_deployment/           # 模型部署示例
│   ├── CMakeLists.txt
│   ├── cnn_deployment.cpp     # CIFAR-10 CNN部署演示
│   ├── README.md              # 部署示例详细说明
│   ├── models/                # 测试模型和图像
│   │   └── test_image.png     # 测试图像
│   └── images/                # 输出图像
├── performance_benchmark/      # 性能基准测试
│   ├── CMakeLists.txt
│   └── performance_test.cpp   # 综合性能测试
└── advanced_features/          # 高级功能示例
    ├── CMakeLists.txt
    ├── jit_demo.cpp           # JIT优化演示
    └── memory_pool_demo.cpp   # 内存池演示
```

## 编译和运行

### 1. 编译所有示例

```bash
cd example
mkdir build
cd build
cmake ..
make -j4
```

### 2. 运行特定示例

```bash
# 基础操作示例
./basic_operations/basic_gemm
./basic_operations/basic_gemv
./basic_operations/basic_dot
./basic_operations/basic_axpy

# 深度学习示例
./deep_learning/relu_demo
./deep_learning/softmax_demo
./deep_learning/conv2d_demo

# 模型推理示例
./model_inference/simple_cnn_inference

# 模型部署示例
./model_deployment/cnn_deployment

# 性能基准测试
./performance_benchmark/performance_test

# 高级功能示例
./advanced_features/jit_demo
./advanced_features/memory_pool_demo
```

### 3. 运行所有示例

```bash
make examples
```

## 示例说明

### 基础操作示例

#### basic_gemm.cpp
- 演示如何使用GEMM进行矩阵乘法
- 展示性能测试和结果验证
- 包含JIT优化演示

#### basic_gemv.cpp
- 演示如何使用GEMV进行矩阵向量乘法
- 展示不同矩阵大小的性能对比
- 包含JIT优化演示

#### basic_dot.cpp
- 演示如何使用DOT计算向量点积
- 展示大向量的性能测试
- 包含JIT优化演示

#### basic_axpy.cpp
- 演示如何使用AXPY进行向量加法
- 展示向量化操作的效果
- 包含JIT优化演示

### 深度学习示例

#### relu_demo.cpp
- 演示如何使用ReLU激活函数
- 展示批量数据处理
- 包含激活率统计

#### softmax_demo.cpp
- 演示如何使用Softmax归一化
- 展示分类任务的概率分布
- 包含结果验证

#### conv2d_demo.cpp
- 演示如何使用Conv2D卷积
- 展示卷积层的性能测试
- 包含输出统计

### 模型推理示例

#### simple_cnn_inference.cpp
- 演示完整的CNN模型推理流程
- 包含卷积层、池化层、ReLU激活
- 展示性能基准测试（928.2 FPS）
- 实现简化的分类网络架构

### 模型部署示例

#### cnn_deployment.cpp
- 完整的CIFAR-10 CNN模型部署
- 集成OpenCV图像预处理
- 3层卷积网络（32→64→128卷积核）
- 高性能推理（170.5 FPS，5.864ms）
- 详细的结果可视化和分析
- 支持自定义图像输入

### 性能基准测试

#### performance_test.cpp
- 综合性能测试程序
- 测试各种算子的性能
- 生成详细的性能报告
- 包含GPU信息显示

### 高级功能示例

#### jit_demo.cpp
- 演示JIT优化的效果
- 对比有无JIT的性能差异
- 展示JIT缓存机制
- 包含JIT统计信息

#### memory_pool_demo.cpp
- 演示内存池的使用
- 展示内存池的性能优势
- 包含碎片化分析
- 提供优化建议

## 性能优化建议

### 1. 使用JIT优化
- 对于重复执行的算子，启用JIT优化
- 利用JIT缓存机制减少编译时间
- 监控JIT统计信息

### 2. 使用内存池
- 对于频繁分配释放的张量，使用内存池
- 监控内存池的碎片化情况
- 根据使用模式调整内存池大小

### 3. 选择合适的算子
- 根据数据大小选择合适的算子实现
- 利用算子的自动优化功能
- 监控算子的性能指标

### 4. 批量处理
- 尽量使用批量处理提高效率
- 合理设置批量大小
- 利用GPU的并行计算能力

### 5. 模型部署优化
- 使用Tensor Core加速（如果支持）
- 实现算子融合减少内存访问
- 优化数据预处理管道
- 使用混合精度计算（FP16/BF16）

### 6. 图像处理优化
- 使用GPU加速的图像预处理
- 实现异步数据传输
- 优化内存布局和数据格式

## 故障排除

### 1. 编译错误
- 确保CUDA环境正确安装
- 检查CMake配置
- 确保所有依赖库已安装

### 2. 运行时错误
- 检查GPU内存是否足够
- 确保CUDA设备可用
- 检查张量形状是否正确
- 验证图像文件路径和格式

### 3. 性能问题
- 使用性能分析工具
- 检查内存使用情况
- 优化数据访问模式
- 监控GPU利用率和内存带宽

### 4. 模型部署问题
- 检查OpenCV是否正确安装
- 验证图像预处理结果
- 确保模型权重正确初始化
- 检查输出张量的形状和数值范围

## 扩展示例

你可以基于这些示例创建自己的应用：

1. 复制现有示例文件
2. 修改张量大小和参数
3. 添加自己的业务逻辑
4. 测试和优化性能

### 模型部署扩展

- **自定义模型**：修改网络架构和参数
- **新数据集**：适配不同的输入格式和类别
- **推理服务**：构建REST API或gRPC服务
- **批处理**：实现批量推理优化
- **模型量化**：集成INT8/FP16量化

### 性能优化扩展

- **Tensor Core**：集成Tensor Core加速
- **算子融合**：实现自定义融合算子
- **内存优化**：优化内存访问模式
- **多GPU**：支持多GPU并行推理

## 快速开始

### 新手推荐路径

1. **基础操作** → `basic_operations/` - 了解基本算子使用
2. **深度学习** → `deep_learning/` - 学习激活函数和卷积
3. **模型推理** → `model_inference/` - 体验完整推理流程
4. **模型部署** → `model_deployment/` - 掌握生产级部署
5. **性能测试** → `performance_benchmark/` - 验证性能表现
6. **高级功能** → `advanced_features/` - 探索优化技术

### 性能基准

| 示例 | 性能指标 | 说明 |
|------|----------|------|
| 简单CNN推理 | 928.2 FPS | 基础推理性能 |
| CIFAR-10部署 | 170.5 FPS | 完整部署性能 |
| GEMM操作 | 886.65 GFLOPS | 矩阵运算性能 |
| ReLU激活 | 88.61 GFLOPS | 激活函数性能 |

## 贡献

欢迎贡献新的示例程序！请确保：

1. 代码风格一致
2. 包含详细的注释
3. 添加适当的错误处理
4. 提供性能测试结果
5. 更新相关文档

