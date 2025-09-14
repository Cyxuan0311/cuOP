#include "cuda_op/detail/cuBlas/gemm.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <type_traits>
#include <cuda_fp16.h>
#include <mma.h>  // 添加WMMA支持

namespace cu_op_mem {

// ==================== Kernel 1: 基础GEMM Kernel (支持转置) ====================
template <typename T>
__global__ void gemm_kernel_basic(int m, int n, int k, T alpha, const T* A, const T* B, T beta, T* C, 
                                 bool transA, bool transB, int lda, int ldb, int ldc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= m || col >= n) return;
    
    T sum = 0;
    
    for (int i = 0; i < k; ++i) {
        T a_val, b_val;
        
        if (transA) {
            a_val = A[i * lda + row];
        } else {
            a_val = A[row * lda + i];
        }
        
        if (transB) {
            b_val = B[col * ldb + i];
        } else {
            b_val = B[i * ldb + col];
        }
        
        sum += a_val * b_val;
    }
    
    C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
}

// ==================== Kernel 2: Tile优化版本 (不转置) ====================
template <typename T, int TILE_SIZE>
__global__ void gemm_kernel_tiled(int m, int n, int k, T alpha, const T* A, const T* B, T beta, T* C) {
    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    T sum = 0;
    
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 加载tile到shared memory
        int tiled_col = t * TILE_SIZE + threadIdx.x;
        int tiled_row = t * TILE_SIZE + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < m && tiled_col < k) ? A[row * k + tiled_col] : 0;
        Bs[threadIdx.y][threadIdx.x] = (col < n && tiled_row < k) ? B[tiled_row * n + col] : 0;
        
        __syncthreads();
        
        // 计算tile内的点积
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < n) {
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}

// ==================== Kernel 3: Warp-level优化版本 ====================
template <typename T, int TILE_SIZE>
__global__ void gemm_kernel_warp_optimized(int m, int n, int k, T alpha, const T* A, const T* B, T beta, T* C) {
    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];
    
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + warp_id;
    int col = blockIdx.x * TILE_SIZE + lane_id;
    
    T sum = 0;
    
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int tiled_col = t * TILE_SIZE + lane_id;
        int tiled_row = t * TILE_SIZE + warp_id;
        
        As[warp_id][lane_id] = (row < m && tiled_col < k) ? A[row * k + tiled_col] : 0;
        Bs[warp_id][lane_id] = (col < n && tiled_row < k) ? B[tiled_row * n + col] : 0;
        
        __syncthreads();
        
        // Warp-level reduction
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[warp_id][i] * Bs[i][lane_id];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < n) {
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}

// ==================== Kernel 4: Tensor Core版本 (集成实现) ====================
__global__ void gemm_kernel_tensor_core_fp16(int m, int n, int k, half alpha, 
                                            const half* A, const half* B, half beta, half* C) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda::wmma;
    
    // 定义tile大小
    const int M_TILES = (m + 15) / 16;
    const int N_TILES = (n + 15) / 16;
    const int K_TILES = (k + 15) / 16;
    
    // 声明fragment
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, half> c_frag;
    
    // 计算当前tile的位置
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    if (warpM >= M_TILES || warpN >= N_TILES) return;
    
    // 初始化accumulator
    fill_fragment(c_frag, 0.0f);
    
    // 主循环：遍历K维度
    for (int i = 0; i < K_TILES; ++i) {
        int aRow = warpM * 16;
        int aCol = i * 16;
        int bRow = i * 16;
        int bCol = warpN * 16;
        
        // 边界检查
        if (aRow < m && aCol < k) {
            load_matrix_sync(a_frag, A + aRow * k + aCol, k);
        } else {
            fill_fragment(a_frag, 0.0f);
        }
        
        if (bRow < k && bCol < n) {
            load_matrix_sync(b_frag, B + bRow * n + bCol, n);
        } else {
            fill_fragment(b_frag, 0.0f);
        }
        
        // Tensor Core计算
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 存储结果
    int cRow = warpM * 16;
    int cCol = warpN * 16;
    
    if (cRow < m && cCol < n) {
        store_matrix_sync(C + cRow * n + cCol, c_frag, n, mem_row_major);
    }
#else
    // 对于不支持Tensor Core的架构，使用简单的实现
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        half sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
#endif
}

// 混合精度Tensor Core GEMM (FP16输入，FP32累加)
__global__ void gemm_kernel_tensor_core_mixed(int m, int n, int k, float alpha, 
                                             const half* A, const half* B, float beta, float* C) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda::wmma;
    
    const int M_TILES = (m + 15) / 16;
    const int N_TILES = (n + 15) / 16;
    const int K_TILES = (k + 15) / 16;
    
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    if (warpM >= M_TILES || warpN >= N_TILES) return;
    
    fill_fragment(c_frag, 0.0f);
    
    for (int i = 0; i < K_TILES; ++i) {
        int aRow = warpM * 16;
        int aCol = i * 16;
        int bRow = i * 16;
        int bCol = warpN * 16;
        
        if (aRow < m && aCol < k) {
            load_matrix_sync(a_frag, A + aRow * k + aCol, k);
        } else {
            fill_fragment(a_frag, 0.0f);
        }
        
        if (bRow < k && bCol < n) {
            load_matrix_sync(b_frag, B + bRow * n + bCol, n);
        } else {
            fill_fragment(b_frag, 0.0f);
        }
        
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    int cRow = warpM * 16;
    int cCol = warpN * 16;
    
    if (cRow < m && cCol < n) {
        store_matrix_sync(C + cRow * n + cCol, c_frag, n, mem_row_major);
    }
#else
    // 对于不支持Tensor Core的架构，使用简单的实现
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += __half2float(A[row * k + i]) * __half2float(B[i * n + col]);
        }
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
#endif
}

// ==================== Kernel 5: 分块优化版本 (大矩阵) ====================
template <typename T, int BLOCK_SIZE>
__global__ void gemm_kernel_blocked(int m, int n, int k, T alpha, const T* A, const T* B, T beta, T* C) {
    __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    T sum = 0;
    
    for (int t = 0; t < (k + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // 协作加载
        if (row < m && t * BLOCK_SIZE + tx < k) {
            As[ty][tx] = A[row * k + t * BLOCK_SIZE + tx];
        } else {
            As[ty][tx] = 0;
        }
        
        if (col < n && t * BLOCK_SIZE + ty < k) {
            Bs[ty][tx] = B[(t * BLOCK_SIZE + ty) * n + col];
        } else {
            Bs[ty][tx] = 0;
        }
        
        __syncthreads();
        
        // 计算
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < n) {
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}

// ==================== Kernel选择器 ====================
enum class GemmKernelType {
    BASIC,
    TILED,
    WARP_OPTIMIZED,
    TENSOR_CORE,
    BLOCKED
};

template <typename T>
struct GemmKernelSelector {
    static GemmKernelType SelectKernel(int m, int n, int k, bool transA, bool transB) {
        // 检查是否支持Tensor Core
        bool tensor_core_supported = false;
        #if __CUDA_ARCH__ >= 700
        if (std::is_same<T, half>::value || std::is_same<T, float>::value) {
            tensor_core_supported = true;
        }
        #endif
        
        // 改进的kernel选择策略
        if (transA || transB) {
            // 转置情况根据矩阵大小选择
            if (m >= 1024 && n >= 1024 && k >= 1024) {
                return GemmKernelType::BLOCKED;
            } else if (m >= 256 && n >= 256 && k >= 256) {
                return GemmKernelType::WARP_OPTIMIZED;
            } else {
                return GemmKernelType::TILED;
            }
        }
        
        // 计算矩阵大小指标
        int matrix_size = m * n * k;
        int min_dim = std::min({m, n, k});
        int max_dim = std::max({m, n, k});
        
        // Tensor Core优先（如果支持且矩阵足够大）
        if (tensor_core_supported && m >= 256 && n >= 256 && k >= 256) {
            // 对于Tensor Core，确保矩阵维度是16的倍数
            if (m % 16 == 0 && n % 16 == 0 && k % 16 == 0) {
                return GemmKernelType::TENSOR_CORE;
            }
        }
        
        // 超大矩阵使用分块优化
        if (matrix_size >= 1024 * 1024 * 1024 || (m >= 2048 && n >= 2048 && k >= 2048)) {
            return GemmKernelType::BLOCKED;
        }
        
        // 大矩阵使用warp优化
        if (matrix_size >= 64 * 1024 * 1024 || (m >= 512 && n >= 512 && k >= 512)) {
            return GemmKernelType::WARP_OPTIMIZED;
        }
        
        // 中等矩阵使用tile优化
        if (matrix_size >= 1024 * 1024 || (m >= 128 && n >= 128 && k >= 128)) {
            return GemmKernelType::TILED;
        }
        
        // 小矩阵使用基础kernel
        return GemmKernelType::BASIC;
    }
    
    // 动态调整block和thread配置
    static void GetOptimalConfig(int m, int n, int k, GemmKernelType kernel_type, 
                                dim3& blocks, dim3& threads) {
        switch (kernel_type) {
            case GemmKernelType::BASIC: {
                threads = dim3(16, 16);
                blocks = dim3((n + 15) / 16, (m + 15) / 16);
                break;
            }
            case GemmKernelType::TILED: {
                constexpr int TILE_SIZE = 16;
                threads = dim3(TILE_SIZE, TILE_SIZE);
                blocks = dim3((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
                break;
            }
            case GemmKernelType::WARP_OPTIMIZED: {
                constexpr int TILE_SIZE = 32;
                threads = dim3(TILE_SIZE, TILE_SIZE);
                blocks = dim3((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
                break;
            }
            case GemmKernelType::BLOCKED: {
                constexpr int BLOCK_SIZE = 32;
                threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
                blocks = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
                break;
            }
            case GemmKernelType::TENSOR_CORE: {
                threads = dim3(32, 8);  // 4 warps per block
                blocks = dim3((n + 15) / 16, (m + 15) / 16);
                break;
            }
        }
    }
};

// ==================== 构造函数和析构函数 ====================
template <typename T>
Gemm<T>::Gemm(bool transA, bool transB, T alpha, T beta)
    : transA_(transA), transB_(transB), alpha_(alpha), beta_(beta) {
    // 初始化cublas handle
    cublasCreate(&handle_);
}

template <typename T>
Gemm<T>::~Gemm() {
    if (handle_) {
        cublasDestroy(handle_);
    }
}

template <typename T>
void Gemm<T>::SetWeight(const Tensor<T>& weight) {
    weight_ = Tensor<T>(weight.shape());
    cudaMemcpy(weight_.data(), weight.data(), weight.bytes(), cudaMemcpyDeviceToDevice);
}

// ==================== 主要Forward函数 ====================
template <typename T>
StatusCode Gemm<T>::Forward(const Tensor<T>& input, Tensor<T>& output) {
    int m = static_cast<int>(input.shape()[0]);
    int k = static_cast<int>(input.shape()[1]);
    int n = static_cast<int>(weight_.shape()[1]);
    
    if (weight_.shape()[0] != k || output.shape()[0] != m || output.shape()[1] != n) {
        LOG(ERROR) << "Gemm: shape mismatch";
        return StatusCode::SHAPE_MISMATCH;
    }
    
    const T* d_A = input.data();
    const T* d_B = weight_.data();
    T* d_C = output.data();
    
    // 选择最优kernel
    GemmKernelType kernel_type = GemmKernelSelector<T>::SelectKernel(m, n, k, transA_, transB_);
    
    cudaError_t err = cudaSuccess;
    dim3 threads, blocks;
    
    // 获取最优配置
    GemmKernelSelector<T>::GetOptimalConfig(m, n, k, kernel_type, blocks, threads);
    
    switch (kernel_type) {
        case GemmKernelType::BASIC: {
            cu_op_mem::gemm_kernel_basic<T><<<blocks, threads>>>(
                m, n, k, alpha_, d_A, d_B, beta_, d_C, 
                transA_, transB_, k, n, n
            );
            break;
        }
        
        case GemmKernelType::TILED: {
            constexpr int TILE_SIZE = 16;
            cu_op_mem::gemm_kernel_tiled<T, TILE_SIZE><<<blocks, threads>>>(
                m, n, k, alpha_, d_A, d_B, beta_, d_C
            );
            break;
        }
        
        case GemmKernelType::WARP_OPTIMIZED: {
            constexpr int TILE_SIZE = 32;  // Warp size
            cu_op_mem::gemm_kernel_warp_optimized<T, TILE_SIZE><<<blocks, threads>>>(
                m, n, k, alpha_, d_A, d_B, beta_, d_C
            );
            break;
        }
        
        case GemmKernelType::BLOCKED: {
            constexpr int BLOCK_SIZE = 32;
            cu_op_mem::gemm_kernel_blocked<T, BLOCK_SIZE><<<blocks, threads>>>(
                m, n, k, alpha_, d_A, d_B, beta_, d_C
            );
            break;
        }
        
        case GemmKernelType::TENSOR_CORE: {
            #if __CUDA_ARCH__ >= 700
            if (std::is_same<T, half>::value) {
                // FP16 Tensor Core
                gemm_kernel_tensor_core_fp16<<<blocks, threads>>>(
                    m, n, k, static_cast<half>(alpha_), 
                    reinterpret_cast<const half*>(d_A), 
                    reinterpret_cast<const half*>(d_B), 
                    static_cast<half>(beta_), 
                    reinterpret_cast<half*>(d_C)
                );
            } else if (std::is_same<T, float>::value) {
                // 混合精度 Tensor Core (需要FP16输入)
                LOG(WARNING) << "Mixed precision Tensor Core requires FP16 input";
                return StatusCode::NOT_IMPLEMENTED;
            } else {
                LOG(WARNING) << "Tensor Core not supported for this data type";
                return StatusCode::NOT_IMPLEMENTED;
            }
            #else
            LOG(WARNING) << "Tensor Core not supported on this GPU architecture";
            return StatusCode::NOT_IMPLEMENTED;
            #endif
            break;
        }
    }
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "gemm kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    
    LOG(INFO) << "Gemm: C = " << alpha_ << " * A * B + " << beta_ << " * C, "
              << "m = " << m << ", n = " << n << ", k = " << k 
              << ", kernel = " << static_cast<int>(kernel_type);
    
    return StatusCode::SUCCESS;
}

// ==================== 显式实例化 ====================
template class Gemm<float>;
template class Gemm<double>;

} // namespace cu_op_mem