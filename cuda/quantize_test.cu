#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "kernel.cuh"
#include "utility.cuh"

static void usage(const char* prog) {
    printf("Usage:\n");
    printf("  %s\n", prog);
    printf("  %s M N K p mp iter\n", prog);
    printf("  %s output.csv M N K p mp iter\n", prog);
    printf("\n");
    printf("M=batch, N=neurons, K=hidden. Defaults: M=64 N=11008 K=4096 p=50 mp=0 iter=10\n");
}

static bool validate_dims(const int batch, const int neurons, const int hidden) {
    if (batch % SPARSITY_GROUP_SIZE != 0) {
        fprintf(stderr, "M(batch) must be a multiple of %d\n", SPARSITY_GROUP_SIZE);
        return false;
    }
    if (batch % 8 != 0) {
        fprintf(stderr, "M(batch) must be a multiple of 8 for the sparse MM tile writeback\n");
        return false;
    }
    if (neurons % 256 != 0) {
        fprintf(stderr, "N(neurons) must be a multiple of 256 for the current sparse MM tile writeback\n");
        return false;
    }
    if (hidden % 256 != 0) {
        fprintf(stderr, "K(hidden) must be a multiple of 256 for the current sparse MM tile writeback\n");
        return false;
    }
    if (hidden % 2 != 0) {
        fprintf(stderr, "K(hidden) must be even for packed INT4 weights\n");
        return false;
    }
    return true;
}

static float rand_float() {
    return 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
}

static void fill_random_float(float* data, const size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = rand_float();
    }
}

static void fill_random_half(half* data, const size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = __float2half(rand_float());
    }
}

static void generate_sparse_idx(float* idx_host, const int batch, const int neurons, const int p, const int mp) {
    for (int row = 0; row < batch; row += SPARSITY_GROUP_SIZE) {
        generateIdx(idx_host + row * neurons, SPARSITY_GROUP_SIZE, neurons, p, mp);
    }
}

static int clamp_int(const int x, const int lo, const int hi) {
    return std::max(lo, std::min(hi, x));
}

static void quantize_per_neuron_reference(const half* src, uint8_t* dst_q4, int8_t* dst_q8,
                                          half* scale_q4, half* scale_q8,
                                          half* dst_deq_q4, half* dst_deq_q8,
                                          const int rows, const int cols) {
    std::fill(dst_q4, dst_q4 + ((size_t)rows * cols) / 2, 0);

    for (int row = 0; row < rows; ++row) {
        const int row_base = row * cols;
        float max_abs = 0.0f;
        for (int col = 0; col < cols; ++col) {
            max_abs = std::max(max_abs, std::fabs(__half2float(src[row_base + col])));
        }

        const float s4 = max_abs == 0.0f ? 0.0f : max_abs / 7.0f;
        const float s8 = max_abs == 0.0f ? 0.0f : max_abs / 127.0f;
        scale_q4[row] = __float2half(s4);
        scale_q8[row] = __float2half(s8);

        const float deq_s4 = __half2float(scale_q4[row]);
        const float deq_s8 = __half2float(scale_q8[row]);
        const float inv_s4 = deq_s4 == 0.0f ? 0.0f : 1.0f / deq_s4;
        const float inv_s8 = deq_s8 == 0.0f ? 0.0f : 1.0f / deq_s8;

        for (int col = 0; col < cols; ++col) {
            const int offset = row_base + col;
            const float x = __half2float(src[offset]);

            const int q4 = clamp_int((int)lrintf(x * inv_s4), -8, 7);
            const uint8_t q4_bits = (uint8_t)(q4 & 0x0F);
            const int q4_byte = offset >> 1;
            if ((offset & 1) == 0) {
                dst_q4[q4_byte] = (uint8_t)((dst_q4[q4_byte] & 0xF0) | q4_bits);
            } else {
                dst_q4[q4_byte] = (uint8_t)((dst_q4[q4_byte] & 0x0F) | (q4_bits << 4));
            }
            dst_deq_q4[offset] = __float2half((float)q4 * deq_s4);

            const int q8 = clamp_int((int)lrintf(x * inv_s8), -128, 127);
            dst_q8[offset] = (int8_t)q8;
            dst_deq_q8[offset] = __float2half((float)q8 * deq_s8);
        }
    }
}

struct CheckResult {
    float max_abs;
    float max_rel;
    size_t mismatches;
    size_t worst;
};

static CheckResult check_close(const char* name, const float* actual_device, const float* ref_device,
                               const size_t size, const float atol = 1e-2f, const float rtol = 1e-2f) {
    std::vector<float> actual(size);
    std::vector<float> ref(size);

    CUDA_CHECK(cudaMemcpy(actual.data(), actual_device, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref.data(), ref_device, size * sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t worst = 0;
    size_t mismatches = 0;

    for (size_t i = 0; i < size; ++i) {
        const float diff = std::fabs(actual[i] - ref[i]);
        const float rel = diff / std::max(1.0f, std::fabs(ref[i]));
        if (diff > max_abs) {
            max_abs = diff;
            max_rel = rel;
            worst = i;
        }
        if (diff > atol + rtol * std::fabs(ref[i])) {
            mismatches++;
        }
    }

    printf("%s: max_abs=%g max_rel=%g mismatches=%zu/%zu", name, max_abs, max_rel, mismatches, size);
    if (mismatches > 0) {
        printf(" worst_idx=%zu actual=%g ref=%g", worst, actual[worst], ref[worst]);
    }
    printf("\n");

    return CheckResult{max_abs, max_rel, mismatches, worst};
}

static CheckResult check_sparse_up_close(const char* name, const float* actual_device, const float* ref_device,
                                         const int neurons, const int batch, const int* act_neurons_device,
                                         const float atol = 1e-2f, const float rtol = 1e-2f) {
    const int groups = batch / SPARSITY_GROUP_SIZE;
    std::vector<int> act_neurons(groups);
    std::vector<float> actual((size_t)batch * neurons);
    std::vector<float> ref((size_t)batch * neurons);

    CUDA_CHECK(cudaMemcpy(act_neurons.data(), act_neurons_device, groups * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actual.data(), actual_device, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref.data(), ref_device, ref.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t worst = 0;
    size_t mismatches = 0;
    size_t checked = 0;

    for (int col = 0; col < batch; ++col) {
        const int group_id = col / SPARSITY_GROUP_SIZE;
        const int actM = group_id == 0 ? act_neurons[group_id] : (act_neurons[group_id] - act_neurons[group_id - 1]);
        for (int row = 0; row < actM; ++row) {
            const size_t idx = (size_t)col * neurons + row;
            const float diff = std::fabs(actual[idx] - ref[idx]);
            const float rel = diff / std::max(1.0f, std::fabs(ref[idx]));
            if (diff > max_abs) {
                max_abs = diff;
                max_rel = rel;
                worst = idx;
            }
            if (diff > atol + rtol * std::fabs(ref[idx])) {
                mismatches++;
            }
            checked++;
        }
    }

    printf("%s: max_abs=%g max_rel=%g mismatches=%zu/%zu", name, max_abs, max_rel, mismatches, checked);
    if (mismatches > 0) {
        printf(" worst_idx=%zu actual=%g ref=%g", worst, actual[worst], ref[worst]);
    }
    printf("\n");

    return CheckResult{max_abs, max_rel, mismatches, worst};
}

template <typename KernelFn>
static float benchmark(const char* name, const int iter, cudaStream_t stream, KernelFn fn) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float time = 0.0f;
    float time_min = 1e9f;

    for (int i = 0; i < iter; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        fn();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));

        printf("%d: %s time: %f ms\n", i, name, time);
        time_min = fminf(time_min, time);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return time_min;
}

static void cublas_dense_up(cublasHandle_t handle, const half* weight, const float* input,
                            half* input_h, float* output,
                            const int neurons, const int batch, const int hidden,
                            cudaStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    convert_fp32_to_fp16_cuda(input, input_h, batch * hidden, stream);
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              neurons, batch, hidden,
                              &alpha,
                              weight, CUDA_R_16F, hidden,
                              input_h, CUDA_R_16F, hidden,
                              &beta,
                              output, CUDA_R_32F, neurons,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
}

static void cublas_dense_down(cublasHandle_t handle, const half* weight, const float* input,
                              half* input_h, float* output,
                              const int hidden, const int batch, const int neurons,
                              cudaStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    convert_fp32_to_fp16_cuda(input, input_h, batch * neurons, stream);
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              hidden, batch, neurons,
                              &alpha,
                              weight, CUDA_R_16F, hidden,
                              input_h, CUDA_R_16F, neurons,
                              &beta,
                              output, CUDA_R_32F, hidden,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
}

int main(int argc, char** argv) {
    srand(0);

    const char* output_file = nullptr;
    int batch = 64;
    int neurons = 11008;
    int hidden = 4096;
    int p = 50;
    int mp = 0;
    int iter = 10;

    if (argc == 7) {
        batch = atoi(argv[1]);
        neurons = atoi(argv[2]);
        hidden = atoi(argv[3]);
        p = atoi(argv[4]);
        mp = atoi(argv[5]);
        iter = atoi(argv[6]);
    } else if (argc == 8) {
        output_file = argv[1];
        batch = atoi(argv[2]);
        neurons = atoi(argv[3]);
        hidden = atoi(argv[4]);
        p = atoi(argv[5]);
        mp = atoi(argv[6]);
        iter = atoi(argv[7]);
    } else if (argc != 1) {
        usage(argv[0]);
        return 1;
    }

    if (!validate_dims(batch, neurons, hidden)) {
        return 1;
    }

    printf("quantize_test: M=%d N=%d K=%d p=%d mp=%d iter=%d\n", batch, neurons, hidden, p, mp, iter);

    const size_t input_size = (size_t)batch * hidden;
    const size_t down_input_size = (size_t)batch * neurons;
    const size_t up_out_size = (size_t)batch * neurons;
    const size_t down_out_size = (size_t)batch * hidden;
    const size_t weight_size = (size_t)neurons * hidden;
    const size_t q4_size = weight_size / 2;
    const int groups = batch / SPARSITY_GROUP_SIZE;

    float* input = nullptr;
    float* down_input = nullptr;
    float* idx = nullptr;
    half* weight_fp16 = nullptr;
    half* weight_deq_q4 = nullptr;
    half* weight_deq_q8 = nullptr;
    uint8_t* weight_q4 = nullptr;
    int8_t* weight_q8 = nullptr;
    half* scale_q4 = nullptr;
    half* scale_q8 = nullptr;
    int* merge_idx = nullptr;
    int* act_neurons_device = nullptr;

    float* up_fp16 = nullptr;
    float* up_ref_q4 = nullptr;
    float* up_ref_q8 = nullptr;
    float* up_q4 = nullptr;
    float* up_q8 = nullptr;
    float* down_fp16 = nullptr;
    float* down_ref_q4 = nullptr;
    float* down_ref_q8 = nullptr;
    float* down_q4 = nullptr;
    float* down_q8 = nullptr;
    half* input_h = nullptr;
    half* down_input_h = nullptr;
    float* up_cublas = nullptr;
    float* down_cublas = nullptr;

    CUDA_CHECK(cudaMalloc(&input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&down_input, down_input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&idx, up_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weight_fp16, weight_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&weight_deq_q4, weight_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&weight_deq_q8, weight_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&weight_q4, q4_size * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&weight_q8, weight_size * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&scale_q4, (size_t)neurons * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&scale_q8, (size_t)neurons * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&merge_idx, (size_t)neurons * groups * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&act_neurons_device, groups * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&up_fp16, up_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&up_ref_q4, up_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&up_ref_q8, up_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&up_q4, up_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&up_q8, up_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&down_fp16, down_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&down_ref_q4, down_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&down_ref_q8, down_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&down_q4, down_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&down_q8, down_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&input_h, input_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&down_input_h, down_input_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&up_cublas, up_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&down_cublas, down_out_size * sizeof(float)));

    float* h_input = nullptr;
    float* h_down_input = nullptr;
    float* h_idx = nullptr;
    half* h_weight_fp16 = nullptr;
    half* h_weight_deq_q4 = nullptr;
    half* h_weight_deq_q8 = nullptr;
    uint8_t* h_weight_q4 = nullptr;
    int8_t* h_weight_q8 = nullptr;
    half* h_scale_q4 = nullptr;
    half* h_scale_q8 = nullptr;

    CUDA_CHECK(cudaMallocHost(&h_input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_down_input, down_input_size * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_idx, up_out_size * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_weight_fp16, weight_size * sizeof(half)));
    CUDA_CHECK(cudaMallocHost(&h_weight_deq_q4, weight_size * sizeof(half)));
    CUDA_CHECK(cudaMallocHost(&h_weight_deq_q8, weight_size * sizeof(half)));
    CUDA_CHECK(cudaMallocHost(&h_weight_q4, q4_size * sizeof(uint8_t)));
    CUDA_CHECK(cudaMallocHost(&h_weight_q8, weight_size * sizeof(int8_t)));
    CUDA_CHECK(cudaMallocHost(&h_scale_q4, (size_t)neurons * sizeof(half)));
    CUDA_CHECK(cudaMallocHost(&h_scale_q8, (size_t)neurons * sizeof(half)));

    fill_random_float(h_input, input_size);
    fill_random_float(h_down_input, down_input_size);
    fill_random_half(h_weight_fp16, weight_size);
    generate_sparse_idx(h_idx, batch, neurons, p, mp);
    quantize_per_neuron_reference(h_weight_fp16, h_weight_q4, h_weight_q8,
                                  h_scale_q4, h_scale_q8,
                                  h_weight_deq_q4, h_weight_deq_q8,
                                  neurons, hidden);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
    CUBLAS_CHECK(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));

    CUDA_CHECK(cudaMemcpyAsync(input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(down_input, h_down_input, down_input_size * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(idx, h_idx, up_out_size * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(weight_fp16, h_weight_fp16, weight_size * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(weight_deq_q4, h_weight_deq_q4, weight_size * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(weight_deq_q8, h_weight_deq_q8, weight_size * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(weight_q4, h_weight_q4, q4_size * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(weight_q8, h_weight_q8, weight_size * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(scale_q4, h_scale_q4, (size_t)neurons * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(scale_q8, h_scale_q8, (size_t)neurons * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    get_idx_cuda(idx, merge_idx, act_neurons_device, batch, neurons, stream);

    resetMatrix<<<(up_out_size + 255) / 256, 256, 0, stream>>>(up_fp16, up_out_size);
    resetMatrix<<<(up_out_size + 255) / 256, 256, 0, stream>>>(up_ref_q4, up_out_size);
    resetMatrix<<<(up_out_size + 255) / 256, 256, 0, stream>>>(up_ref_q8, up_out_size);
    resetMatrix<<<(up_out_size + 255) / 256, 256, 0, stream>>>(up_q4, up_out_size);
    resetMatrix<<<(up_out_size + 255) / 256, 256, 0, stream>>>(up_q8, up_out_size);
    resetMatrix<<<(down_out_size + 255) / 256, 256, 0, stream>>>(down_fp16, down_out_size);
    resetMatrix<<<(down_out_size + 255) / 256, 256, 0, stream>>>(down_ref_q4, down_out_size);
    resetMatrix<<<(down_out_size + 255) / 256, 256, 0, stream>>>(down_ref_q8, down_out_size);
    resetMatrix<<<(down_out_size + 255) / 256, 256, 0, stream>>>(down_q4, down_out_size);
    resetMatrix<<<(down_out_size + 255) / 256, 256, 0, stream>>>(down_q8, down_out_size);

    up_mul_mat_cuda_sparse(weight_fp16, input, up_fp16, merge_idx, neurons, batch, hidden, act_neurons_device, stream);
    up_mul_mat_cuda_sparse(weight_deq_q4, input, up_ref_q4, merge_idx, neurons, batch, hidden, act_neurons_device, stream);
    up_mul_mat_cuda_sparse(weight_deq_q8, input, up_ref_q8, merge_idx, neurons, batch, hidden, act_neurons_device, stream);
    dequantize_mm_up_cuda_sparse(weight_q4, scale_q4, input, up_q4, merge_idx, neurons, batch, hidden, act_neurons_device, 4, stream);
    dequantize_mm_up_cuda_sparse(weight_q8, scale_q8, input, up_q8, merge_idx, neurons, batch, hidden, act_neurons_device, 8, stream);

    down_mul_mat_cuda_sparse(weight_fp16, down_input, down_fp16, merge_idx, hidden, batch, neurons, act_neurons_device, stream);
    down_mul_mat_cuda_sparse(weight_deq_q4, down_input, down_ref_q4, merge_idx, hidden, batch, neurons, act_neurons_device, stream);
    down_mul_mat_cuda_sparse(weight_deq_q8, down_input, down_ref_q8, merge_idx, hidden, batch, neurons, act_neurons_device, stream);
    dequantize_mm_down_cuda_sparse(weight_q4, scale_q4, down_input, down_q4, merge_idx, hidden, batch, neurons, act_neurons_device, 4, stream);
    dequantize_mm_down_cuda_sparse(weight_q8, scale_q8, down_input, down_q8, merge_idx, hidden, batch, neurons, act_neurons_device, 8, stream);

    cublas_dense_up(cublas_handle, weight_fp16, input, input_h, up_cublas, neurons, batch, hidden, stream);
    cublas_dense_down(cublas_handle, weight_fp16, down_input, down_input_h, down_cublas, hidden, batch, neurons, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    CheckResult up_q4_check = check_sparse_up_close("up int4 kernel vs fp16 dequant ref", up_q4, up_ref_q4, neurons, batch, act_neurons_device);
    CheckResult up_q8_check = check_sparse_up_close("up int8 kernel vs fp16 dequant ref", up_q8, up_ref_q8, neurons, batch, act_neurons_device);
    CheckResult down_q4_check = check_close("down int4 kernel vs fp16 dequant ref", down_q4, down_ref_q4, down_out_size);
    CheckResult down_q8_check = check_close("down int8 kernel vs fp16 dequant ref", down_q8, down_ref_q8, down_out_size);

    check_sparse_up_close("up int4 quantization drift vs original fp16", up_ref_q4, up_fp16, neurons, batch, act_neurons_device, 1e9f, 1e9f);
    check_sparse_up_close("up int8 quantization drift vs original fp16", up_ref_q8, up_fp16, neurons, batch, act_neurons_device, 1e9f, 1e9f);
    check_close("down int4 quantization drift vs original fp16", down_ref_q4, down_fp16, down_out_size, 1e9f, 1e9f);
    check_close("down int8 quantization drift vs original fp16", down_ref_q8, down_fp16, down_out_size, 1e9f, 1e9f);

    float up_fp16_ms = benchmark("mm_up fp16", iter, stream, [&]() {
        resetMatrix<<<(up_out_size + 255) / 256, 256, 0, stream>>>(up_fp16, up_out_size);
        up_mul_mat_cuda_sparse(weight_fp16, input, up_fp16, merge_idx, neurons, batch, hidden, act_neurons_device, stream);
    });

    float up_q4_ms = benchmark("dequantize_mm_up int4", iter, stream, [&]() {
        resetMatrix<<<(up_out_size + 255) / 256, 256, 0, stream>>>(up_q4, up_out_size);
        dequantize_mm_up_cuda_sparse(weight_q4, scale_q4, input, up_q4, merge_idx, neurons, batch, hidden, act_neurons_device, 4, stream);
    });

    float up_q8_ms = benchmark("dequantize_mm_up int8", iter, stream, [&]() {
        resetMatrix<<<(up_out_size + 255) / 256, 256, 0, stream>>>(up_q8, up_out_size);
        dequantize_mm_up_cuda_sparse(weight_q8, scale_q8, input, up_q8, merge_idx, neurons, batch, hidden, act_neurons_device, 8, stream);
    });

    float down_fp16_ms = benchmark("mm_down fp16", iter, stream, [&]() {
        resetMatrix<<<(down_out_size + 255) / 256, 256, 0, stream>>>(down_fp16, down_out_size);
        down_mul_mat_cuda_sparse(weight_fp16, down_input, down_fp16, merge_idx, hidden, batch, neurons, act_neurons_device, stream);
    });

    float down_q4_ms = benchmark("dequantize_mm_down int4", iter, stream, [&]() {
        resetMatrix<<<(down_out_size + 255) / 256, 256, 0, stream>>>(down_q4, down_out_size);
        dequantize_mm_down_cuda_sparse(weight_q4, scale_q4, down_input, down_q4, merge_idx, hidden, batch, neurons, act_neurons_device, 4, stream);
    });

    float down_q8_ms = benchmark("dequantize_mm_down int8", iter, stream, [&]() {
        resetMatrix<<<(down_out_size + 255) / 256, 256, 0, stream>>>(down_q8, down_out_size);
        dequantize_mm_down_cuda_sparse(weight_q8, scale_q8, down_input, down_q8, merge_idx, hidden, batch, neurons, act_neurons_device, 8, stream);
    });

    float cublas_up_fp16_ms = benchmark("cublas_dense_up fp16", iter, stream, [&]() {
        cublas_dense_up(cublas_handle, weight_fp16, input, input_h, up_cublas, neurons, batch, hidden, stream);
    });

    float cublas_up_q4_ms = benchmark("cublas_dense_up deq_int4", iter, stream, [&]() {
        cublas_dense_up(cublas_handle, weight_deq_q4, input, input_h, up_cublas, neurons, batch, hidden, stream);
    });

    float cublas_up_q8_ms = benchmark("cublas_dense_up deq_int8", iter, stream, [&]() {
        cublas_dense_up(cublas_handle, weight_deq_q8, input, input_h, up_cublas, neurons, batch, hidden, stream);
    });

    float cublas_down_fp16_ms = benchmark("cublas_dense_down fp16", iter, stream, [&]() {
        cublas_dense_down(cublas_handle, weight_fp16, down_input, down_input_h, down_cublas, hidden, batch, neurons, stream);
    });

    float cublas_down_q4_ms = benchmark("cublas_dense_down deq_int4", iter, stream, [&]() {
        cublas_dense_down(cublas_handle, weight_deq_q4, down_input, down_input_h, down_cublas, hidden, batch, neurons, stream);
    });

    float cublas_down_q8_ms = benchmark("cublas_dense_down deq_int8", iter, stream, [&]() {
        cublas_dense_down(cublas_handle, weight_deq_q8, down_input, down_input_h, down_cublas, hidden, batch, neurons, stream);
    });

    printf("best ms: up_fp16=%f up_int4=%f up_int8=%f down_fp16=%f down_int4=%f down_int8=%f "
           "cublas_up_fp16=%f cublas_up_deq_int4=%f cublas_up_deq_int8=%f "
           "cublas_down_fp16=%f cublas_down_deq_int4=%f cublas_down_deq_int8=%f\n",
           up_fp16_ms, up_q4_ms, up_q8_ms, down_fp16_ms, down_q4_ms, down_q8_ms,
           cublas_up_fp16_ms, cublas_up_q4_ms, cublas_up_q8_ms,
           cublas_down_fp16_ms, cublas_down_q4_ms, cublas_down_q8_ms);

    if (output_file != nullptr) {
        FILE* f = fopen(output_file, "a+");
        if (f == nullptr) {
            fprintf(stderr, "Can not open output file: %s\n", output_file);
        } else {
            fprintf(f, "%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%g,%g,%zu,%g,%g,%zu,%g,%g,%zu,%g,%g,%zu\n",
                    batch, neurons, hidden, p, mp,
                    up_fp16_ms, up_q4_ms, up_q8_ms, down_fp16_ms, down_q4_ms, down_q8_ms,
                    cublas_up_fp16_ms, cublas_up_q4_ms, cublas_up_q8_ms,
                    cublas_down_fp16_ms, cublas_down_q4_ms, cublas_down_q8_ms,
                    up_q4_check.max_abs, up_q4_check.max_rel, up_q4_check.mismatches,
                    up_q8_check.max_abs, up_q8_check.max_rel, up_q8_check.mismatches,
                    down_q4_check.max_abs, down_q4_check.max_rel, down_q4_check.mismatches,
                    down_q8_check.max_abs, down_q8_check.max_rel, down_q8_check.mismatches);
            fclose(f);
        }
    }

    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(down_input));
    CUDA_CHECK(cudaFree(idx));
    CUDA_CHECK(cudaFree(weight_fp16));
    CUDA_CHECK(cudaFree(weight_deq_q4));
    CUDA_CHECK(cudaFree(weight_deq_q8));
    CUDA_CHECK(cudaFree(weight_q4));
    CUDA_CHECK(cudaFree(weight_q8));
    CUDA_CHECK(cudaFree(scale_q4));
    CUDA_CHECK(cudaFree(scale_q8));
    CUDA_CHECK(cudaFree(merge_idx));
    CUDA_CHECK(cudaFree(act_neurons_device));
    CUDA_CHECK(cudaFree(up_fp16));
    CUDA_CHECK(cudaFree(up_ref_q4));
    CUDA_CHECK(cudaFree(up_ref_q8));
    CUDA_CHECK(cudaFree(up_q4));
    CUDA_CHECK(cudaFree(up_q8));
    CUDA_CHECK(cudaFree(down_fp16));
    CUDA_CHECK(cudaFree(down_ref_q4));
    CUDA_CHECK(cudaFree(down_ref_q8));
    CUDA_CHECK(cudaFree(down_q4));
    CUDA_CHECK(cudaFree(down_q8));
    CUDA_CHECK(cudaFree(input_h));
    CUDA_CHECK(cudaFree(down_input_h));
    CUDA_CHECK(cudaFree(up_cublas));
    CUDA_CHECK(cudaFree(down_cublas));

    CUDA_CHECK(cudaFreeHost(h_input));
    CUDA_CHECK(cudaFreeHost(h_down_input));
    CUDA_CHECK(cudaFreeHost(h_idx));
    CUDA_CHECK(cudaFreeHost(h_weight_fp16));
    CUDA_CHECK(cudaFreeHost(h_weight_deq_q4));
    CUDA_CHECK(cudaFreeHost(h_weight_deq_q8));
    CUDA_CHECK(cudaFreeHost(h_weight_q4));
    CUDA_CHECK(cudaFreeHost(h_weight_q8));
    CUDA_CHECK(cudaFreeHost(h_scale_q4));
    CUDA_CHECK(cudaFreeHost(h_scale_q8));

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
