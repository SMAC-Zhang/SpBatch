#include <bits/types/FILE.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "utility.cuh"
#include "kernel.cuh"

size_t M = 8;
const size_t N = 11008;
const size_t K = 4096;
const bool correctness = true;
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

void MLP_PowerInfer(const float* input, const half* up_w, const half* down_w, const half* gate_w, float* idx, float**dst, cudaStream_t stream) {
    // gate
    {
        const int nrows = N;
        const int ncols = K;
        const int src1_ncols = M;
        const int dst_ne0 = N;
        convert_mul_mat_batch_f16_cuda_sparse((void*)gate_w, input, dst[0], ncols, nrows, src1_ncols, dst_ne0, stream, nullptr, idx);
    }
    // up
    {
        const int nrows = N;
        const int ncols = K;
        const int src1_ncols = M;
        const int dst_ne0 = N;
        convert_mul_mat_batch_f16_cuda_sparse((void*)up_w, input, dst[1], ncols, nrows, src1_ncols, dst_ne0, stream, nullptr, idx);
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
        const int ncols = K;
        const int nrows = N;
        const int src1_rows = N;
        const int src1_ncols = M;
        convert_axpy_sparse_batch_f16_cuda((void*)down_w, dst[3], dst[4], ncols, nrows, src1_rows, src1_ncols, stream, NULL, idx);
    }
}

void MLP_BS(const float* input, const half* up_w, const half* down_w, const half* gate_w, float* idx, float**dst, cudaStream_t stream0, cudaStream_t stream1) {
    const int groups = M / SPARSITY_GROUP_SIZE;
    static int * merge_idx = nullptr, *act_neurons_device = nullptr;;
    if (merge_idx == nullptr) {
        CUDA_CHECK(cudaMalloc(&merge_idx, N * groups * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&act_neurons_device, groups * sizeof(int)));
    }

    get_idx_cuda(idx, merge_idx, act_neurons_device, M, N, stream0);
    gate_and_up_mul_mat_cuda_sparse(gate_w, up_w, input, dst[0], dst[1], merge_idx, N, M, K, act_neurons_device, stream0, stream1);
    relu_and_mul_cuda(dst[0], dst[1], dst[3], M, N, stream0);

    down_mul_mat_cuda_sparse(down_w, dst[3], dst[4], merge_idx, K, M, N, act_neurons_device, stream0);
}

void MLP_naive(const float* input, const half* up_w, const half* down_w, const half* gate_w, float* idx, float**dst, float* dst_down, cudaStream_t stream0, cudaStream_t stream1) {
    const int groups = M / SPARSITY_GROUP_SIZE;
    static int * merge_idx = nullptr, *act_neurons_device = nullptr;;
    if (merge_idx == nullptr) {
        CUDA_CHECK(cudaMalloc(&merge_idx, N * groups * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&act_neurons_device, groups * sizeof(int)));
    }

    get_idx_cuda(idx, merge_idx, act_neurons_device, M, N, stream0);
    // gate
    naive_mm_up_cuda(gate_w, input, dst[0], N, M, K, act_neurons_device, merge_idx, stream0);
    // up
    naive_mm_up_cuda(up_w, input, dst[1], N, M, K, act_neurons_device, merge_idx, stream0);
    // act
    relu_f32_cuda(dst[0], dst[2], M * N, stream0);
    // gate @ up
    mul_f32_cuda(dst[2], dst[1], dst[3], M * N, M * N, stream0);
    // down
    naive_mm_down_cuda(down_w, dst[3], dst_down, K, M, N, act_neurons_device, merge_idx, stream0);
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

    randFill<float>(h_input, M * K);
    randFill<half>(h_up_w, K * N);
    randFill<half>(h_gate_w, K * N);
    randFill<half>(h_down_w, N * K);
    read_idx(layer, N, M, h_idx_h);

    for (int i = 0; i < M * N; ++i) {
        h_idx[i] = (float)h_idx_h[i];
    }

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


    // cuda core PowerInfer
    {
        float time_min = 1e9;
        for (int i = 0; i < iter; ++i) {
            CUDA_CHECK(cudaEventRecord(start, stream0));

            MLP_PowerInfer(input, up_w, down_w, gate_w, idx, dst, stream0);

            CUDA_CHECK(cudaEventRecord(stop, stream0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));

            printf("%d: PowerInfer time: %f ms\n", i, time);
            time_min = fmin(time_min, time);
        }
        fprintf(f, "%f,", time_min);
    }

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

        // BS
    {
        cudaDeviceSynchronize();
        
        cudaEvent_t start0, start1, stop0, stop1;
        CUDA_CHECK(cudaEventCreate(&start0));
        CUDA_CHECK(cudaEventCreate(&start1));
        CUDA_CHECK(cudaEventCreate(&stop0));
        CUDA_CHECK(cudaEventCreate(&stop1));
        
        float time0 = 0.0f, time1 = 0.0f;
        float time_min = 1e9;

        for (int i = 0; i < iter; ++i) {
            CUDA_CHECK(cudaEventRecord(start0, stream0));
            CUDA_CHECK(cudaEventRecord(start1, stream1));

            MLP_BS(input, up_w, down_w, gate_w, idx, dst, stream0, stream1);

            CUDA_CHECK(cudaEventRecord(stop0, stream0));
            CUDA_CHECK(cudaEventRecord(stop1, stream1));
            CUDA_CHECK(cudaEventSynchronize(stop0));
            CUDA_CHECK(cudaEventSynchronize(stop1));
            CUDA_CHECK(cudaGetLastError());
            
            CUDA_CHECK(cudaEventElapsedTime(&time0, start0, stop0));
            CUDA_CHECK(cudaEventElapsedTime(&time1, start1, stop1));
            
            printf("%d: gate time: %f ms, up time: %f ms\n", i, time0, time1);
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
        if (correctness) {
            cudaDeviceSynchronize();
            float * dst_ref = nullptr;
            CUDA_CHECK(cudaMalloc(&dst_ref, M * K * sizeof(float)));
            
            MLP_naive(input, up_w, down_w, gate_w, idx, dst, dst_ref, stream0, stream1);

            cudaDeviceSynchronize();
            CUDA_CHECK(cudaGetLastError());

            compare_cuda(dst[4], dst_ref, M, K, stream0);
            cudaDeviceSynchronize();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaFree(dst_ref));
        }
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