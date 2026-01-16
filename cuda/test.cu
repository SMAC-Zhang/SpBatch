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
#include "kernel.cuh"

size_t M = 8;
const size_t N = 16384;
const size_t K = 4096;
const bool correctness = false;
// const int layer = 15;
// const int iter = 1;
// const bool test_correctness = false;

void MLP_llamacpp(const float* input, const half* up_w, const half* down_w, const half* gate_w, float**dst, cudaStream_t stream, cublasHandle_t handle) {
    // gate
    {
        static half* input_h = nullptr, *dst_h = nullptr;
        if (input_h == nullptr) {
            CUDA_CHECK(cudaMalloc(&input_h, M * K * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&dst_h, M * N * sizeof(half)));
        }

        convert_fp32_to_fp16_cuda(input, input_h, M * K, stream);

        const half alpha = 1.0f;
        const half beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, \
            N, M, K, &alpha, gate_w, CUDA_R_16F, K, \
            input_h, CUDA_R_16F, K, &beta, \
            dst_h, CUDA_R_16F, N, \
            CUBLAS_COMPUTE_16F, \
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        convert_fp16_to_fp32_cuda(dst_h, dst[0], M * N, stream);
    }
    // up
    {
        static half* input_h = nullptr, *dst_h = nullptr;
        if (input_h == nullptr) {
            CUDA_CHECK(cudaMalloc(&input_h, M * K * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&dst_h, M * N * sizeof(half)));
        }

        convert_fp32_to_fp16_cuda(input, input_h, M * K, stream);

        const half alpha = 1.0f;
        const half beta = 0.0f;
        // CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, \
            N, M, K, &alpha, up_w, CUDA_R_16F, K, \
            input_h, CUDA_R_16F, K, &beta, \
            dst_h, CUDA_R_16F, N, \
            CUBLAS_COMPUTE_16F, \
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        convert_fp16_to_fp32_cuda(dst_h, dst[1], M * N, stream);
    }
    // act
    {
        relu_f32_cuda(dst[0], dst[2], M * N, stream);
    }
    // gate @ up
    {
        mul_f32_cuda(dst[2], dst[1], dst[3], M * N, M * N, stream);
    }
    // down
    {
        half* input_h = nullptr, *dst_h = nullptr;
        if (input_h == nullptr) {
            CUDA_CHECK(cudaMalloc(&input_h, M * N * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&dst_h, M * K * sizeof(half)));
        }

        convert_fp32_to_fp16_cuda(dst[3], input_h, M * N, stream);

        const half alpha = 1.0f;
        const half beta = 0.0f;
        // CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, \
            K, M, N, &alpha, down_w, CUDA_R_16F, K, \
            input_h, CUDA_R_16F, N , &beta, \
            dst_h, CUDA_R_16F, K, \
            CUBLAS_COMPUTE_16F, \
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        convert_fp16_to_fp32_cuda(dst_h, dst[4], M * K, stream);
    }
}

int main(int argc, char** argv) {
    srand(0);

    char* filename = argv[1];
    M = atoi(argv[2]);
    int layer = atoi(argv[3]);
    int iter = atoi(argv[4]);
    FILE *f = fopen(filename, "a+");
    fprintf(f, "%d,%ld,", layer, M);
    printf("***********layer: %d****************\n", layer);

    float *input, *idx;
    half *up_w, *gate_w, *down_w;
    float *dst[5];

    CUDA_CHECK(cudaMalloc(&input, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&idx, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&up_w, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&gate_w, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&down_w, N * K * sizeof(half)));

    for (int i = 0; i <= 3; ++i) {
        CUDA_CHECK(cudaMalloc(&dst[i], M * N * sizeof(float)));
    }
    CUDA_CHECK(cudaMalloc(&dst[4], M * K * sizeof(float)));

    // host pinned allocations
    float*   h_input    = nullptr;
    float*   h_idx      = nullptr;
    half*    h_idx_h = nullptr;
    half*    h_up_w     = nullptr;
    half*    h_gate_w   = nullptr;
    half*    h_down_w   = nullptr;

    CUDA_CHECK(cudaMallocHost(&h_input,    M * K * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_idx,      M * N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_idx_h,    M * N * sizeof(half)));
    CUDA_CHECK(cudaMallocHost(&h_up_w,     K * N * sizeof(half)));
    CUDA_CHECK(cudaMallocHost(&h_gate_w,   K * N * sizeof(half)));
    CUDA_CHECK(cudaMallocHost(&h_down_w,   N * K * sizeof(half)));


    // copy to device
    CUDA_CHECK(cudaMemcpy(input, h_input, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(up_w, h_up_w, K * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gate_w, h_gate_w, K * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(down_w, h_down_w, N * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(idx, h_idx, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // record events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float time = 0.0f;

    // create streams
    cudaStream_t stream0, stream1;
    CUDA_CHECK(cudaStreamCreate(&stream0));
    CUDA_CHECK(cudaStreamCreate(&stream1));

    // llama.cpp
    {
        float time_min = 1e9;
        cudaDeviceSynchronize();
        cublasHandle_t handle;      
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
        for (int i = 0; i < iter; ++i) {

            CUDA_CHECK(cudaEventRecord(start));

            MLP_llamacpp(input, up_w, down_w, gate_w, dst, 0, handle);

            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
            
            printf("%d: cublas time: %f ms\n", i, time);
            time_min = fmin(time_min, time);
        }
        fprintf(f, "%f,", time_min);
        cublasDestroy(handle);
    }


    // release resources
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(idx));
    CUDA_CHECK(cudaFree(up_w));
    CUDA_CHECK(cudaFree(gate_w));
    CUDA_CHECK(cudaFree(down_w));
    for (int i = 0; i <= 4; ++i) {
        CUDA_CHECK(cudaFree(dst[i]));
    }
    CUDA_CHECK(cudaFreeHost(h_input));
    CUDA_CHECK(cudaFreeHost(h_idx));
    CUDA_CHECK(cudaFreeHost(h_up_w));
    CUDA_CHECK(cudaFreeHost(h_gate_w));
    CUDA_CHECK(cudaFreeHost(h_down_w));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    CUDA_CHECK(cudaStreamDestroy(stream0));
    CUDA_CHECK(cudaStreamDestroy(stream1));

    return 0;
}
