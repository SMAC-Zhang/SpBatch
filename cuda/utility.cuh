#pragma once

#include <cstddef>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

void read_idx(const int layer, const int hidden_size, const int tokens, void * buffer);

void randIdx(float * idx, size_t size, int p);

void generateIdx(float *idx, int M, int N, int p, int mp);

void compare_cuda(float * a, float * b, int M, int N, cudaStream_t stream);

template<class T>
void randFill(T * data, size_t size) {
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<T>(dist(gen));
        if (rand() % 2 == 0) {
            data[i] = -data[i];
        }
        // data[i] = 1.0f;
    }
}

template<class T>
__global__ void printMatrix(T * data, size_t size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	printf("%f ", static_cast<float>(data[idx]));
}

template<class T>
__global__ void resetMatrix(T * data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    data[idx] = static_cast<T>(0.f);
}