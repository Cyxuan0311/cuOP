#!/usr/bin/env python3
"""
cuOP Python Package Setup

This setup script configures the Python package for cuOP, a high-performance
CUDA operator and memory management library with JIT optimization.
"""

import os
import sys
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from pybind11.setup_helpers import Pybind11Extension, build_ext

# 检查Python版本
if sys.version_info < (3, 7):
    raise RuntimeError("Python 3.7+ is required")

# 获取项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

# 获取版本信息
def get_version():
    """从CMakeLists.txt或环境变量获取版本号"""
    version = os.environ.get('CUOP_VERSION', '0.1.0')
    return version

# 检查CUDA环境
def check_cuda():
    """检查CUDA环境是否可用"""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    
    # 检查环境变量
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home and os.path.exists(cuda_home):
        return True
    
    return False

# 获取CUDA信息
def get_cuda_info():
    """获取CUDA版本和路径信息"""
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if not cuda_home:
        return None, None
    
    # 尝试获取CUDA版本
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            # 解析版本号
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    version = line.split('release')[1].strip().split(',')[0]
                    return cuda_home, version
    except FileNotFoundError:
        pass
    
    return cuda_home, None

# 配置编译选项
def get_compile_args():
    """获取编译选项"""
    args = [
        '-std=c++17',
        '-O3',
        '-DNDEBUG',
        '-fPIC',
        '-Wall',
        '-Wextra',
        '-Wno-unused-parameter',
        '-Wno-unused-variable',
    ]
    
    # 添加CUDA相关选项
    if check_cuda():
        args.extend([
            '-DCUDA_ENABLED',
            '-D__CUDA_ARCH__=89',  # RTX 4050
        ])
    
    return args

def get_link_args():
    """获取链接选项"""
    args = []
    
    if check_cuda():
        args.extend([
            '-lcuda',
            '-lcudart',
            '-lcublas',
            '-lcudnn',
            '-lnvrtc',
        ])
    
    return args

# 定义扩展模块
ext_modules = [
    Pybind11Extension(
        "cuop.core",
        [
            "cuop/core.cpp",
            "cuop/tensor.cpp",
            "cuop/operators.cpp",
            "cuop/jit.cpp",
            "cuop/memory_pool.cpp",
            # cuBlas算子源文件
            "../src/cuda_op/detail/cuBlas/scal.cu",
            "../src/cuda_op/detail/cuBlas/axpy.cu",
            "../src/cuda_op/detail/cuBlas/copy.cu",
            "../src/cuda_op/detail/cuBlas/dot.cu",
            "../src/cuda_op/detail/cuBlas/gemm.cu",
            "../src/cuda_op/detail/cuBlas/gemv.cu",
            "../src/cuda_op/detail/cuBlas/symm.cu",
            "../src/cuda_op/detail/cuBlas/trsm.cu",
            # cuDNN算子源文件
            "../src/cuda_op/detail/cuDNN/relu.cu",
            "../src/cuda_op/detail/cuDNN/softmax.cu",
            "../src/cuda_op/detail/cuDNN/batchnorm.cu",
            "../src/cuda_op/detail/cuDNN/layernorm.cu",
            "../src/cuda_op/detail/cuDNN/convolution.cu",
            "../src/cuda_op/detail/cuDNN/matmul.cu",
            "../src/cuda_op/detail/cuDNN/batchmatmul.cu",
            "../src/cuda_op/detail/cuDNN/flatten.cu",
            "../src/cuda_op/detail/cuDNN/view.cu",
            "../src/cuda_op/detail/cuDNN/maxpool.cu",
            "../src/cuda_op/detail/cuDNN/averagepool.cu",
            "../src/cuda_op/detail/cuDNN/globalmaxpool.cu",
            "../src/cuda_op/detail/cuDNN/globalaverpool.cu",
            # 工具源文件
            "../src/util/status_code.cpp",
        ],
        include_dirs=[
            "../include",
            "../third_party/pybind11/include",
            "../third_party/eigen",
            "/usr/local/cuda/include",
        ],
        extra_compile_args=get_compile_args(),
        extra_link_args=get_link_args(),
        language='c++',
    ),
]

# 包配置
setup(
    name="cuop",
    version=get_version(),
    author="cuOP Team",
    author_email="contact@cuop.dev",
    description="High-performance CUDA operator and memory management library with JIT optimization",
    long_description=open("../README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cuop/cuop",
    project_urls={
        "Bug Tracker": "https://github.com/cuop/cuop/issues",
        "Documentation": "https://cuop.readthedocs.io/",
        "Source Code": "https://github.com/cuop/cuop",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pybind11>=2.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "examples": [
            "matplotlib>=3.3.0",
            "jupyter>=1.0",
            "ipywidgets>=7.6",
        ],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    include_package_data=True,
    package_data={
        "cuop": ["*.pyi", "py.typed"],
    },
    entry_points={
        "console_scripts": [
            "cuop-benchmark=cuop.cli.benchmark:main",
            "cuop-test=cuop.cli.test:main",
        ],
    },
) 