#pragma once

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>
#include <cassert>
#include <sys/types.h>
#include <cub/cub.cuh>

#define CUDA_CHECK(err)                                                                 \
    do {                                                                                \
        cudaError_t err_ = (err);                                                       \
        if (err_ != cudaSuccess) {                                                      \
            int id;                                                                     \
            cudaGetDevice(&id);                                                         \
            fprintf(stderr, "\nCUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, \
                cudaGetErrorString(err_));                                              \
            fprintf(stderr, "current device: %d\n", id);                                \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

#define CUBLAS_CHECK(err)                                                               \
    do {                                                                                \
        cublasStatus_t err_ = (err);                                                    \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                            \
            int id;                                                                     \
            cudaGetDevice(&id);                                                         \
            fprintf(stderr, "\ncuBLAS error %d at %s:%d: %s\n",                         \
                    err_, __FILE__, __LINE__, cublasGetStatusString(err_));             \
            fprintf(stderr, "current device: %d\n", id);                                \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define GGML_CUDA_DMMV_X 32
#define WARP_SIZE 32
#define GGML_CUDA_MMV_Y 1
#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define CUDA_DEQUANTIZE_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_MUL_BLOCK_SIZE 256

// Sparse division related
#define CUDA_RELU_MUL_BLOCK_SIZE 256
#define SPARSITY_GROUP_SIZE 8
#define MAX_GROUPS 64

// WMMA API parameters
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define SWIZZLE_UNIT 8
#define HALF_PER_FLOAT4 8


typedef float dfloat; // dequantize float
typedef float2 dfloat2;
typedef void (*dequantize_kernel_t)(const void * vx, const int ib, const int iqs, dfloat2 & v);

const float dev_sparse_threshold = 0.0f;

// ROW_WARPS * WARP_ROW_TENSORS must be multiple of 4
template<int ROW_WARPS_, int COL_WARPS_, int WARP_ROW_TENSORS_, int WARP_COL_TENSORS_, int BLOCK_K_TENSORS_>
struct SparseMMConfig {
    // block size
    static constexpr int ROW_WARPS = ROW_WARPS_;
    static constexpr int COL_WARPS = COL_WARPS_;
    static constexpr int BLOCK_WARPS = ROW_WARPS * COL_WARPS;

    static constexpr int WARP_ROW_TENSORS = WARP_ROW_TENSORS_;
    static constexpr int WARP_COL_TENSORS = WARP_COL_TENSORS_;
    static constexpr int BLOCK_K_TENSORS = BLOCK_K_TENSORS_; // multiple of 4

    // double buffer shared memory size
    // 2 * BM * BK * sizeof(half) + 2 * BN * BK * sizeof(half)
    static constexpr int BM = MMA_M * ROW_WARPS * WARP_ROW_TENSORS;
    static constexpr int BN = MMA_N * COL_WARPS * WARP_COL_TENSORS;
    static constexpr int BK = MMA_K * BLOCK_K_TENSORS;
};

void convert_fp32_to_fp16_cuda(const void * vx, half * y, const int k, cudaStream_t stream);
void convert_fp16_to_fp32_cuda(const void * vx, float * y, const int k, cudaStream_t stream);
void convert_mul_mat_batch_f16_cuda_sparse(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, int src1_ncols, int dst_ne0, cudaStream_t stream, int *lst, float *idx);
void convert_axpy_sparse_batch_f16_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, int src1_rows, int src1_ncols, cudaStream_t stream, int *lst, float *idx);
void relu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream);
void mul_f32_cuda(const float * x, const float * y, float * dst, const int kx, const int ky, cudaStream_t stream);

// parameters:
// M: batch size
// N: neurons
void get_idx_cuda(const float * idx, int * merge_idx, int * act_neurons_device, const int M, const int N, cudaStream_t stream0);

// parameters:
// M: neurons
// N: batch size
// K: embedding size
void up_mul_mat_cuda_sparse(const half* weight, const float* input, float* dst, int * merge_idx, const int M, const int N, const int K, const int * act_neurons_device, cudaStream_t stream);

// parameters:
// M: neurons
// N: batch size
// K: embedding size
void gate_and_up_mul_mat_cuda_sparse(const half* gate_w, const half* up_w, const float* input, float* gate_dst, float* up_dst, int * merge_idx, const int M, const int N, const int K, const int * act_neurons_device, cudaStream_t stream0, cudaStream_t stream1);

// parameters:
// M: batch size
// N: neurons
void relu_and_mul_cuda(const float * gate, const float* up, float * dst, const int M, const int N, cudaStream_t stream);
void relu_relu_mul_cuda(const float * gate, const float* up, float * dst, const int M, const int N, cudaStream_t stream);

// parameters:
// M: embedding size
// N: batch size
// K: neurons
void down_mul_mat_cuda_sparse(const half* weight, const float* input, float* dst, int * merge_idx, const int M, const int N, const int K, const int * act_neurons_device, cudaStream_t stream);

void add_sparse_cuda(const float* a, const half* b, float* c, const int M, const int N, const int * act_neurons_device, const int * merge_idx, cudaStream_t stream);
/*******************test gemm *************************/
void gemm_cuda(const half* a, const float* b, float* c, const int M, const int N, const int K, cudaStream_t stream);
void naive_mm_up_cuda(const half* a, const float* b, float* c, const int M, const int N, const int K, const int * act_neurons_device, int * merge_idx, cudaStream_t stream);
void naive_mm_down_cuda(const half* a, const float* b, float* c, const int M, const int N, const int K, const int * act_neurons_device, int * merge_idx, cudaStream_t stream);
void naive_sparsity_div(const float* idx, int * merge_idx, int* act_neurons, int M, int N);