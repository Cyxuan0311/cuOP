#!/bin/bash

# cuOP 快速构建脚本
# 简化版本，用于日常开发

set -e

echo "🚀 cuOP 快速构建开始..."

# 清理并重新构建
echo "📁 清理构建目录..."
rm -rf build

echo "⚙️  配置CMake..."
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

echo "🔨 开始编译..."
cmake --build build --parallel $(nproc)

echo "✅ 构建完成！"
echo ""
echo "📋 生成的文件:"
find build -name "*.so" -o -name "test_*" | head -5

echo ""
echo "🎯 可用的测试程序:"
find build -name "test_*" -type f -executable | sed 's/^/  /'
