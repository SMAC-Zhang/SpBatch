
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "utility.cuh"
#include "kernel.cuh"

size_t M = 64;
size_t N = 11008;
size_t K = 4096;
const bool test_down = false;
// const int layer = 15;
// const int iter = 1;

void kernel_llamacpp(const half* weight, const float* input, float* dst, cudaStream_t stream, cublasHandle_t handle) {
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
		M, N, K, &alpha, input_h, CUDA_R_16F, K, \
		weight, CUDA_R_16F, K, &beta, \
		dst_h, CUDA_R_16F, M, \
		CUBLAS_COMPUTE_16F, \
		CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    convert_fp16_to_fp32_cuda(dst_h, dst, M * N, stream);

}

void kernel_PowerInfer(const float* input, const half* weight, float* idx, float* dst, cudaStream_t stream) {
    const int nrows = N;
    const int ncols = K;
    const int src1_ncols = M;
    const int dst_ne0 = N;
    convert_mul_mat_batch_f16_cuda_sparse((void*)weight, input, dst, ncols, nrows, src1_ncols, dst_ne0, stream, nullptr, idx);
}

void kernel_BS(const half* weight, const float* input, const float* idx, float* dst, cudaStream_t stream0, cudaStream_t stream1) {
    const int groups = M / SPARSITY_GROUP_SIZE;
    static int * merge_idx = nullptr, *act_neurons_device = nullptr;
    if (merge_idx == nullptr) {
        CUDA_CHECK(cudaMalloc(&merge_idx, N * groups * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&act_neurons_device, groups * sizeof(int)));
    }

    get_idx_cuda(idx, merge_idx, act_neurons_device, M, N, stream0);
    up_mul_mat_cuda_sparse(weight, input, dst, merge_idx, N, M, K, act_neurons_device, stream0);
}

void kernel_BS_down(const half* weight, const float* input, const float* idx, float* dst, cudaStream_t stream0, cudaStream_t stream1) {
    const int groups = M / SPARSITY_GROUP_SIZE;
    static int * merge_idx = nullptr, *act_neurons_device = nullptr;
    if (merge_idx == nullptr) {
        CUDA_CHECK(cudaMalloc(&merge_idx, N * groups * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&act_neurons_device, groups * sizeof(int)));
    }

    get_idx_cuda(idx, merge_idx, act_neurons_device, M, N, stream0);
    down_mul_mat_cuda_sparse(weight, input, dst, merge_idx, K, M, N, act_neurons_device, stream0);
}

void kernel_naive(const float* input, const half* weight, const float* idx, float* dst, cudaStream_t stream0, cudaStream_t stream1) {
    const int groups = M / SPARSITY_GROUP_SIZE;
    static int * merge_idx = nullptr, *act_neurons_device = nullptr;
    if (merge_idx == nullptr) {
        CUDA_CHECK(cudaMalloc(&merge_idx, N * groups * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&act_neurons_device, groups * sizeof(int)));
    }

    get_idx_cuda(idx, merge_idx, act_neurons_device, M, N, stream0);
    naive_mm_up_cuda(weight, input, dst, N, M, K, act_neurons_device, merge_idx, stream0);
}   

void kernel_naive_down(const float* input, const half* weight, const float* idx, float* dst, cudaStream_t stream0, cudaStream_t stream1) {
    const int groups = M / SPARSITY_GROUP_SIZE;
    static int * merge_idx = nullptr, *act_neurons_device = nullptr;
    if (merge_idx == nullptr) {
        CUDA_CHECK(cudaMalloc(&merge_idx, N * groups * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&act_neurons_device, groups * sizeof(int)));
    }

    get_idx_cuda(idx, merge_idx, act_neurons_device, M, N, stream0);
    naive_mm_down_cuda(weight, input, dst, K, M, N, act_neurons_device, merge_idx, stream0);
}

int main(int argc, char** argv) {
    srand(0);

    char* filename = argv[1];
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    K = atoi(argv[4]);
    int p = atoi(argv[5]);
    int mp = atoi(argv[6]);
    int iter = atoi(argv[7]);
    FILE *f = fopen(filename, "a+");
    fprintf(f, "%ld,%ld,%ld,%d,%d,", M, N, K, p, mp);

    float *input = nullptr, *idx = nullptr, *idx_host = nullptr;
    half *weight = nullptr;
    float *dst[3] = {nullptr};
    float *dst_down = nullptr;
    CUDA_CHECK(cudaMalloc(&input, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&idx, M * N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&idx_host, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weight, K * N * sizeof(half)));

    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaMalloc(&dst[i], M * N * sizeof(float)));
    }
    CUDA_CHECK(cudaMalloc(&dst_down, M * K * sizeof(float)));

    // host pinned allocations
    float *h_input = nullptr;
    half *h_weight = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_input,    M * K * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_weight,   K * N * sizeof(half)));

    randFill<float>(h_input, M * K);
    randFill<half>(h_weight, K * N);

    for (size_t i = 0; i < M; i += 8) {
        generateIdx(idx_host + i * N, 8, N, p, mp);
    }

    // copy to device
    CUDA_CHECK(cudaMemcpy(input, h_input, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(idx, idx_host, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weight, h_weight, K * N * sizeof(half), cudaMemcpyHostToDevice));

    // record events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float time = 0.0f;

    // create streams
    cudaStream_t stream0, stream1;
    CUDA_CHECK(cudaStreamCreate(&stream0));
    CUDA_CHECK(cudaStreamCreate(&stream1));


    // cuda core PowerInfer
    {
        float time_min = 1e9;
        for (int i = 0; i < iter; ++i) {
            CUDA_CHECK(cudaEventRecord(start, stream0));

            kernel_PowerInfer(input, weight, idx, dst[0], stream0);

            CUDA_CHECK(cudaEventRecord(stop, stream0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));

            printf("%d: PowerInfer time: %f ms\n", i, time);
            time_min = fmin(time_min, time);
        }
        fprintf(f, "%f,", time_min);
    }

    // cublas
    {
        float time_min = 1e9;
        cudaDeviceSynchronize();
        cublasHandle_t handle;      
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
        for (int i = 0; i < iter; ++i) {

            CUDA_CHECK(cudaEventRecord(start));

            kernel_llamacpp(weight, input, dst[1], 0, handle);

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

    // BS
    {
        float time_min = 1e9;
        cudaDeviceSynchronize();
        cudaEvent_t start0, start1, stop0, stop1;
        CUDA_CHECK(cudaEventCreate(&start0));
        CUDA_CHECK(cudaEventCreate(&start1));
        CUDA_CHECK(cudaEventCreate(&stop0));
        CUDA_CHECK(cudaEventCreate(&stop1));
        
        float time0 = 0.0f, time1 = 0.0f;

        for (int i = 0; i < iter; ++i) {
            CUDA_CHECK(cudaEventRecord(start0, stream0));
            CUDA_CHECK(cudaEventRecord(start1, stream1));

            kernel_BS(weight, input, idx, dst[2], stream0, stream1);

            CUDA_CHECK(cudaEventRecord(stop0, stream0));
            CUDA_CHECK(cudaEventRecord(stop1, stream1));
            CUDA_CHECK(cudaEventSynchronize(stop0));
            CUDA_CHECK(cudaEventSynchronize(stop1));
            CUDA_CHECK(cudaGetLastError());
            
            CUDA_CHECK(cudaEventElapsedTime(&time0, start0, stop0));
            CUDA_CHECK(cudaEventElapsedTime(&time1, start1, stop1));
            
            printf("%d: up mm time: %f ms, idx time: %f ms\n", i, time0, time1);
            time_min = fmin(time_min, fmax(time0, time1));
        }

        fprintf(f, "%f\n", time_min);

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaEventDestroy(start0));
        CUDA_CHECK(cudaEventDestroy(start1));
        CUDA_CHECK(cudaEventDestroy(stop0));
        CUDA_CHECK(cudaEventDestroy(stop1));
    }

    // test correctness
    {
        cudaDeviceSynchronize();
        float * dst_ref = nullptr;
        CUDA_CHECK(cudaMalloc(&dst_ref, M * N * sizeof(float)));
        
        kernel_naive(input, weight, idx, dst_ref, stream0, stream1);

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        compare_cuda(dst[2], dst_ref, M, N, stream0);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(dst_ref));
    }


    if (!test_down) {
        goto release;
    }

    // resetMatrix<float><<<M * N / 256, 256>>>(dst[2], M * N);
    { // down mm test
        cudaDeviceSynchronize();
        
        cudaEvent_t start0, start1, stop0, stop1;
        CUDA_CHECK(cudaEventCreate(&start0));
        CUDA_CHECK(cudaEventCreate(&start1));
        CUDA_CHECK(cudaEventCreate(&stop0));
        CUDA_CHECK(cudaEventCreate(&stop1));
        
        float time0 = 0.0f, time1 = 0.0f;

        for (int i = 0; i < iter; ++i) {
            CUDA_CHECK(cudaEventRecord(start0, stream0));
            CUDA_CHECK(cudaEventRecord(start1, stream1));

            kernel_BS_down(weight, dst[2], idx, dst_down, stream0, stream1);

            CUDA_CHECK(cudaEventRecord(stop0, stream0));
            CUDA_CHECK(cudaEventRecord(stop1, stream1));
            CUDA_CHECK(cudaEventSynchronize(stop0));
            CUDA_CHECK(cudaEventSynchronize(stop1));
            CUDA_CHECK(cudaGetLastError());
            
            CUDA_CHECK(cudaEventElapsedTime(&time0, start0, stop0));
            CUDA_CHECK(cudaEventElapsedTime(&time1, start1, stop1));
            
            printf("%d: down mm time: %f ms, idx time: %f ms\n", i, time0, time1);
            
        }
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaEventDestroy(start0));
        CUDA_CHECK(cudaEventDestroy(start1));
        CUDA_CHECK(cudaEventDestroy(stop0));
        CUDA_CHECK(cudaEventDestroy(stop1));
    }

    {// test correctness
        cudaDeviceSynchronize();
        float * dst_ref = nullptr;
        CUDA_CHECK(cudaMalloc(&dst_ref, M * K * sizeof(float)));
        
        kernel_naive_down(dst[2], weight, idx, dst_ref, stream0, stream1);

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        compare_cuda(dst_down, dst_ref, M, K, stream0);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(dst_ref));
    }

release:
    // release resources
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(idx));
    CUDA_CHECK(cudaFree(weight));
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaFree(dst[i]));
    }
    CUDA_CHECK(cudaFreeHost(idx_host));

    CUDA_CHECK(cudaFreeHost(h_input));
    CUDA_CHECK(cudaFreeHost(h_weight));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    CUDA_CHECK(cudaStreamDestroy(stream0));
    CUDA_CHECK(cudaStreamDestroy(stream1));

    return 0;
}
