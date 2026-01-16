#include <cstddef>
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdlib>
#include <curand_kernel.h>

#include "kernel.cuh"
#include "utility.cuh"

const int M = 11008;
const int N = 64;
const int K = 4096;
const int iter = 10;

void cublasMM(const half* a, const float* b, float* c, cudaStream_t stream, cublasHandle_t handle) {
    static half* b_h = nullptr, *c_h = nullptr;
    if (b_h == nullptr) {
        CUDA_CHECK(cudaMalloc(&b_h, N * K * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&c_h, M * N * sizeof(half)));
    }

    convert_fp32_to_fp16_cuda(b, b_h, N * K, stream);

    const half alpha = 1.0f;
    const half beta = 0.0f;
    CUBLAS_CHECK(cublasSetStream(handle, stream));
	CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, \
		N, M, K, &alpha, b_h, CUDA_R_16F, K, \
		a, CUDA_R_16F, K, &beta, \
		c_h, CUDA_R_16F, N, \
		CUBLAS_COMPUTE_16F, \
		CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    convert_fp16_to_fp32_cuda(c_h, c, M * N, stream);
}

void myMM(const half* a, const float* b, float* c, cudaStream_t stream) {
	gemm_cuda(a, b, c, M, N, K, stream);
}

int main() {
	half * a;
	float *b, * c, * d;
	CUDA_CHECK(cudaMalloc(&a, M * K * sizeof(half)));
	CUDA_CHECK(cudaMalloc(&b, K * N * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&c, M * N * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d, M * N * sizeof(float)));

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));
	cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float time = 0.0f;

	half * a_host;
	float *b_host;
	CUDA_CHECK(cudaMallocHost(&a_host, M * K * sizeof(half)));
	CUDA_CHECK(cudaMallocHost(&b_host, K * N * sizeof(float)));
	randFill<half>(a_host, M * K);
	randFill<float>(b_host, K * N);
	CUDA_CHECK(cudaMemcpyAsync(a, a_host, M * K * sizeof(half), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(b, b_host, K * N * sizeof(float), cudaMemcpyHostToDevice, stream));

	CUDA_CHECK(cudaDeviceSynchronize());


	{ // myMM
		for (int i = 0; i < iter; ++i) {
			CUDA_CHECK(cudaEventRecord(start, stream));
			
			myMM(a, b, c, stream);

			CUDA_CHECK(cudaEventRecord(stop, stream));
			CUDA_CHECK(cudaEventSynchronize(stop));
			CUDA_CHECK(cudaGetLastError());
			CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));

			printf("%d: mygemm time: %f ms\n", i, time);
		}
	}

	{ // cublasMM
        CUDA_CHECK(cudaDeviceSynchronize());
        cublasHandle_t handle;      
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
        for (int i = 0; i < iter; ++i) {
			resetMatrix<<<(M * N + 31) / 32, 32, 0, stream>>>(d, M * N);
            CUDA_CHECK(cudaEventRecord(start, stream));

			cublasMM(a, b, d, stream, handle);

            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
            
            printf("%d: cublas time: %f ms\n", i, time);
        }
		CUDA_CHECK(cudaDeviceSynchronize());
        cublasDestroy(handle);
    }

	{ // test correctness
		compare_cuda(c, d, M, N, stream);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	CUDA_CHECK(cudaFree(a));
	CUDA_CHECK(cudaFree(b));
	CUDA_CHECK(cudaFree(c));
	CUDA_CHECK(cudaFree(d));
	CUDA_CHECK(cudaFreeHost(a_host));
	CUDA_CHECK(cudaFreeHost(b_host));

	CUDA_CHECK(cudaStreamDestroy(stream));
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	return 0;
}