#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "utility.cuh"
#include "kernel.cuh"

const size_t M = 64;
const size_t N = 11008;
const int groups = M / SPARSITY_GROUP_SIZE;
const int layer = 15;
const int iter = 10;

void sparsity_div(const float* idx, int * merge_idx, int * act_neurons, cudaStream_t stream0, cudaStream_t stream1) {
    get_idx_cuda(idx, merge_idx, act_neurons, M, N, stream0);
}

int main() {
    float * idx = nullptr, * idx_host = nullptr;
    half * idx_half = nullptr;
    int * merge_idx = nullptr, * merge_idx_host = nullptr, *merge_idx_device = nullptr;
    int * act_neurons = nullptr, * act_neurons_host = nullptr, * act_neurons_device = nullptr;

    CUDA_CHECK(cudaMalloc(&idx, sizeof(float) * M * N));
    CUDA_CHECK(cudaMallocHost(&idx_host, sizeof(float) * M * N));
    CUDA_CHECK(cudaMallocHost(&idx_half, sizeof(half) * M * N));
    CUDA_CHECK(cudaMalloc(&merge_idx, sizeof(int) * N * groups));
    CUDA_CHECK(cudaMallocHost(&merge_idx_host, sizeof(int) * N * groups));
    CUDA_CHECK(cudaMallocHost(&merge_idx_device, sizeof(int) * N * groups));
    CUDA_CHECK(cudaMalloc(&act_neurons, sizeof(int) * groups));
    CUDA_CHECK(cudaMallocHost(&act_neurons_host, sizeof(int) * groups));
    CUDA_CHECK(cudaMallocHost(&act_neurons_device, sizeof(int) * groups));

    read_idx(layer, N, M, (void*)idx_half);
    for (size_t i = 0; i < M * N; ++i) {
        idx_host[i] = static_cast<float>(idx_half[i]);
    }

    CUDA_CHECK(cudaMemcpy(idx, idx_host, sizeof(float) * M * N, cudaMemcpyHostToDevice));
    
    cudaStream_t stream0, stream1;
    CUDA_CHECK(cudaStreamCreate(&stream0));
    CUDA_CHECK(cudaStreamCreate(&stream1));
    cudaEvent_t start0, start1, stop0, stop1;
    CUDA_CHECK(cudaEventCreate(&start0));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop0));
    CUDA_CHECK(cudaEventCreate(&stop1));
    float time0 = 0.0f, time1 = 0.0f;

    {
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < iter; ++i) {
            CUDA_CHECK(cudaEventRecord(start0, stream0));
            CUDA_CHECK(cudaEventRecord(start1, stream1));

            sparsity_div(idx, merge_idx, act_neurons, stream0, stream1);

            CUDA_CHECK(cudaEventRecord(stop0, stream0));
            CUDA_CHECK(cudaEventRecord(stop1, stream1));
            CUDA_CHECK(cudaEventSynchronize(stop0));
            CUDA_CHECK(cudaEventSynchronize(stop1));
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaEventElapsedTime(&time0, start0, stop0));
            CUDA_CHECK(cudaEventElapsedTime(&time1, start1, stop1));
            
            printf("%d: idx time: %f ms\n", i, std::max(time0, time1));
        }
    }

    { // test correctness
        naive_sparsity_div(idx_host, merge_idx_host, act_neurons_host, M, N);
        CUDA_CHECK(cudaMemcpy(merge_idx_device, merge_idx, sizeof(int) * N * groups, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(act_neurons_device, act_neurons, sizeof(int) * groups, cudaMemcpyDeviceToHost));
        for (int i = 0; i < groups; ++i) {
            int act_device = act_neurons_device[i] - (i == 0 ? 0 : act_neurons_device[i - 1]);
            if (act_neurons_host[i] != act_device) {
                printf("mismatch at group %d: %d vs %d\n", i, act_neurons_host[i], act_device);
            }
        }
        for (int i = 0; i < groups; ++i) {
            printf("group %d: act_neurons: %d\n", i, act_neurons_host[i]);
        }
        for (int i = 0; i < groups; ++i) {
            for (int j = 0; j < act_neurons_host[i]; ++j) {
                if (merge_idx_host[i * N + j] != merge_idx_device[i * N + j]) {
                    printf("mismatch at (%d, %d): %d vs %d\n", i, j, merge_idx_host[i * N + j], merge_idx_device[i * N + j]);
                }
            }
        }
    }

    // release resources
    CUDA_CHECK(cudaFree(idx));
    CUDA_CHECK(cudaFreeHost(idx_host));
    CUDA_CHECK(cudaFree(merge_idx));
    CUDA_CHECK(cudaFreeHost(merge_idx_host));
    CUDA_CHECK(cudaFree(act_neurons));
    CUDA_CHECK(cudaFreeHost(act_neurons_host));

    CUDA_CHECK(cudaStreamDestroy(stream0));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaEventDestroy(start0));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop0));
    CUDA_CHECK(cudaEventDestroy(stop1));
    
    return 0;
}