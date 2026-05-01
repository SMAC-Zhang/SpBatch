#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kernel.cuh"
#include "utility.cuh"

/************************* Q4_0 host helpers *************************/

static float rand_float() {
    return 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
}

static void fill_random_float(float* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = rand_float();
    }
}

static void fill_random_half(half* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = __float2half(rand_float());
    }
}

static void quantize_q4_0_row(const float* src, block_q4_0* dst, int n) {
    const int nb = n / QK4_0;
    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK4_0; ++j) {
            amax = fmaxf(amax, fabsf(src[i * QK4_0 + j]));
        }
        if (amax == 0.0f) {
            dst[i].d = __float2half(0.0f);
            memset(dst[i].qs, 0, QK4_0 / 2);
        } else {
            float d = amax / 7.0f;
            dst[i].d = __float2half(d);
            float id = 1.0f / d;
            for (int j = 0; j < QK4_0 / 2; ++j) {
                int v0 = (int)(roundf(src[i * QK4_0 + j] * id)) + 8;
                int v1 = (int)(roundf(src[i * QK4_0 + j + QK4_0 / 2] * id)) + 8;
                v0 = std::min(std::max(v0, 0), 15);
                v1 = std::min(std::max(v1, 0), 15);
                dst[i].qs[j] = (uint8_t)(v0 | (v1 << 4));
            }
        }
    }
}

static void quantize_q4_0(const float* src, block_q4_0* dst, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        quantize_q4_0_row(src + r * cols, dst + r * (cols / QK4_0), cols);
    }
}

static void dequantize_q4_0_host(const block_q4_0* src, half* dst, int n) {
    const int nb = n / QK4_0;
    for (int i = 0; i < nb; ++i) {
        float d = __half2float(src[i].d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            int vui = src[i].qs[j];
            dst[i * QK4_0 + j] = __float2half(((vui & 0xF) - 8) * d);
            dst[i * QK4_0 + j + QK4_0 / 2] = __float2half(((vui >> 4) - 8) * d);
        }
    }
}

/************************* FFN layer config *************************/

static const int LAYERS = 32;
// static const int N = 11008; // intermediate size (neurons)
// static const int K = 4096;  // hidden size

/************************* Main *************************/

int main(int argc, char** argv) {
    srand(0);

    int M = 8;       // batch size (tokens)
    int N = 11008;   // intermediate size
    int K = 4096;    // hidden size
    int iter = 5;    // iterations per layer

    const char *csv_name = nullptr;
    if (argc >= 2) csv_name = argv[1];
    if (argc >= 3) M = atoi(argv[2]);
    if (argc >= 4) iter = atoi(argv[3]);
    if (argc >= 5) N = atoi(argv[4]);
    if (argc >= 6) K = atoi(argv[5]);

    if (M % SPARSITY_GROUP_SIZE != 0) {
        fprintf(stderr, "M must be a multiple of %d\n", SPARSITY_GROUP_SIZE);
        return 1;
    }

    const int groups = M / SPARSITY_GROUP_SIZE;
    const int gate_blocks = N * (K / QK4_0);
    const int up_blocks   = N * (K / QK4_0);
    const int down_blocks = N * (K / QK4_0); // stored as [N][K] row-major

    printf("dequantize_test: M=%d N=%d K=%d iter=%d layers=%d\n", M, N, K, iter, LAYERS);

    // accumulators across all layers
    float total_powerinfer     = 0.0f;
    float total_cublas_dequant = 0.0f;
    float total_cublas_gemm    = 0.0f;
    float total_spbatch_idx     = 0.0f;
    float total_spbatch_dequant= 0.0f;
    float total_spbatch_mm     = 0.0f;
    float total_spbatch        = 0.0f;

    // device buffers reused across layers
    float   *d_input    = nullptr; // M*K
    float   *d_idx      = nullptr; // M*N
    block_q4_0 *d_gate = nullptr;
    block_q4_0 *d_up   = nullptr;
    block_q4_0 *d_down = nullptr;
    half    *d_gate_fp16  = nullptr; // N*K
    half    *d_up_fp16    = nullptr; // N*K
    half    *d_down_fp16  = nullptr; // N*K
    float   *d_gate_out  = nullptr; // M*N
    float   *d_up_out    = nullptr; // M*N
    float   *d_gate_act  = nullptr; // M*N
    float   *d_gateup    = nullptr; // M*N
    float   *d_output    = nullptr; // M*K
    int     *d_merge_idx = nullptr;
    int     *d_act_neurons = nullptr;
    int     *d_unique_idx = nullptr;
    int     *d_unique_neurons = nullptr;
    half    *d_input_h   = nullptr; // M*K
    half    *d_gateup_h  = nullptr; // M*N

    cudaStream_t stream;
    cublasHandle_t handle;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

#define DEVALLOC(ptr, sz) do { \
    if ((ptr) == nullptr) CUDA_CHECK(cudaMalloc(&(ptr), (sz))); \
} while(0)

    for (int layer = 0; layer < LAYERS; ++layer) {
        printf("========== Layer %d / %d ==========\n", layer, LAYERS);

        // ----- host data -----
        std::vector<half>   h_idx_h(M * N);
        std::vector<float>  h_idx(M * N);
        read_idx(layer, N, M, h_idx_h.data());
        for (int i = 0; i < M * N; ++i) h_idx[i] = (float)h_idx_h[i];

        std::vector<float> h_input(M * K);
        fill_random_float(h_input.data(), M * K);

        // generate random Q4_0 weights
        std::vector<block_q4_0> h_gate(gate_blocks);
        std::vector<block_q4_0> h_up(up_blocks);
        std::vector<block_q4_0> h_down(down_blocks);
        std::vector<float> h_tmp(std::max(N * K, K * N));

        fill_random_float(h_tmp.data(), (size_t)N * K);
        quantize_q4_0(h_tmp.data(), h_gate.data(), N, K);

        fill_random_float(h_tmp.data(), (size_t)N * K);
        quantize_q4_0(h_tmp.data(), h_up.data(), N, K);

        fill_random_float(h_tmp.data(), (size_t)N * K);
        quantize_q4_0(h_tmp.data(), h_down.data(), N, K); // down is [N][K] too

        // dequant to fp16 on host for cublas ref
        std::vector<half> h_gate_fp16((size_t)N * K);
        std::vector<half> h_up_fp16((size_t)N * K);
        std::vector<half> h_down_fp16((size_t)N * K);
        dequantize_q4_0_host(h_gate.data(), h_gate_fp16.data(), N * K);
        dequantize_q4_0_host(h_up.data(),   h_up_fp16.data(),   N * K);
        dequantize_q4_0_host(h_down.data(), h_down_fp16.data(), N * K);

        // ----- device allocations -----
        DEVALLOC(d_input,        (size_t)M * K * sizeof(float));
        DEVALLOC(d_idx,          (size_t)M * N * sizeof(float));
        DEVALLOC(d_gate,         (size_t)gate_blocks * sizeof(block_q4_0));
        DEVALLOC(d_up,           (size_t)up_blocks * sizeof(block_q4_0));
        DEVALLOC(d_down,         (size_t)down_blocks * sizeof(block_q4_0));
        DEVALLOC(d_gate_fp16,    (size_t)N * K * sizeof(half));
        DEVALLOC(d_up_fp16,      (size_t)N * K * sizeof(half));
        DEVALLOC(d_down_fp16,    (size_t)N * K * sizeof(half));
        DEVALLOC(d_gate_out,     (size_t)M * N * sizeof(float));
        DEVALLOC(d_up_out,       (size_t)M * N * sizeof(float));
        DEVALLOC(d_gate_act,     (size_t)M * N * sizeof(float));
        DEVALLOC(d_gateup,       (size_t)M * N * sizeof(float));
        DEVALLOC(d_output,       (size_t)M * K * sizeof(float));
        DEVALLOC(d_merge_idx,    (size_t)N * groups * sizeof(int));
        DEVALLOC(d_act_neurons,  (size_t)groups * sizeof(int));
        DEVALLOC(d_unique_idx,   (size_t)N * sizeof(int));
        DEVALLOC(d_unique_neurons, sizeof(int));
        DEVALLOC(d_input_h,      (size_t)M * K * sizeof(half));
        DEVALLOC(d_gateup_h,     (size_t)M * N * sizeof(half));

        CUDA_CHECK(cudaMemcpyAsync(d_input, h_input.data(), (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_idx,   h_idx.data(),   (size_t)M * N * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_gate,  h_gate.data(),  (size_t)gate_blocks * sizeof(block_q4_0), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_up,    h_up.data(),    (size_t)up_blocks * sizeof(block_q4_0), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_down,  h_down.data(),  (size_t)down_blocks * sizeof(block_q4_0), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_gate_fp16, h_gate_fp16.data(), (size_t)N * K * sizeof(half), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_up_fp16,   h_up_fp16.data(),   (size_t)N * K * sizeof(half), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_down_fp16, h_down_fp16.data(), (size_t)N * K * sizeof(half), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // ----- events -----
        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
        float t = 0.0f;

        // ================================================================
        //  1. PowerInfer  (fused Q4_0 dequant + sparse matmul)
        // ================================================================
        {
            float best = 1e9f;
            for (int i = 0; i < iter; ++i) {
                CUDA_CHECK(cudaEventRecord(ev_start, stream));

                dequantize_mul_mat_batch_q4_0_cuda_sparse(
                    d_gate, d_input, d_gate_out, K, N, M, N, stream, nullptr, d_idx);
                dequantize_mul_mat_batch_q4_0_cuda_sparse(
                    d_up,   d_input, d_up_out,   K, N, M, N, stream, nullptr, d_idx);
                relu_f32_cuda(d_gate_out, d_gate_act, M * N, stream);
                mul_f32_cuda(d_gate_act, d_up_out, d_gateup, M * N, M * N, stream);
                dequantize_axpy_sparse_batch_q4_0_cuda(
                    d_down, d_gateup, d_output, K, N, N, M, stream, nullptr, d_idx);

                CUDA_CHECK(cudaEventRecord(ev_stop, stream));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventElapsedTime(&t, ev_start, ev_stop));
                printf("  PowerInfer[%d]: %f ms\n", i, t);
                best = fminf(best, t);
            }
            total_powerinfer += best;
            printf("  -> PowerInfer best: %f ms\n", best);
        }

        // ================================================================
        //  2. Cublas (full dequant Q4_0 → fp16, then dense gemm)
        // ================================================================
        {
            // --- dequant time ---
            float dequant_best = 1e9f;
            for (int i = 0; i < iter; ++i) {
                CUDA_CHECK(cudaEventRecord(ev_start, stream));
                dequantize_q4_0_full_cuda(d_gate, d_gate_fp16, N * K, stream);
                dequantize_q4_0_full_cuda(d_up,   d_up_fp16,   N * K, stream);
                dequantize_q4_0_full_cuda(d_down, d_down_fp16, N * K, stream);
                CUDA_CHECK(cudaEventRecord(ev_stop, stream));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventElapsedTime(&t, ev_start, ev_stop));
                dequant_best = fminf(dequant_best, t);
            }
            total_cublas_dequant += dequant_best;

            // --- gemm time ---
            float gemm_best = 1e9f;
            const float alpha = 1.0f, beta = 0.0f;
            for (int i = 0; i < iter; ++i) {
                CUDA_CHECK(cudaEventRecord(ev_start, stream));

                convert_fp32_to_fp16_cuda(d_input, d_input_h, M * K, stream);

                // gate gemm: output[M][N] = input[M][K] @ weight[N][K]^T
                CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    d_gate_fp16, CUDA_R_16F, K,
                    d_input_h,   CUDA_R_16F, K,
                    &beta,
                    d_gate_out, CUDA_R_32F, N,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

                // up gemm
                CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    d_up_fp16, CUDA_R_16F, K,
                    d_input_h, CUDA_R_16F, K,
                    &beta,
                    d_up_out, CUDA_R_32F, N,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

                // activation + element-wise
                relu_f32_cuda(d_gate_out, d_gate_act, M * N, stream);
                mul_f32_cuda(d_gate_act, d_up_out, d_gateup, M * N, M * N, stream);

                // down gemm: output[M][K] = gateup[M][N] @ weight[N][K]
                convert_fp32_to_fp16_cuda(d_gateup, d_gateup_h, M * N, stream);
                CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    K, M, N, &alpha,
                    d_down_fp16, CUDA_R_16F, K,
                    d_gateup_h,  CUDA_R_16F, N,
                    &beta,
                    d_output, CUDA_R_32F, K,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

                CUDA_CHECK(cudaEventRecord(ev_stop, stream));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventElapsedTime(&t, ev_start, ev_stop));
                gemm_best = fminf(gemm_best, t);
            }
            total_cublas_gemm += gemm_best;
            printf("  Cublas: dequant best=%f ms  gemm best=%f ms\n", dequant_best, gemm_best);
        }

        // ================================================================
        //  3. SpBatch  (idx + sparse dequant Q4_0 → fp16, then mm_up/mm_down)
        // ================================================================
        {
            cudaEvent_t ev_idx_stop, ev_dequant_stop;
            CUDA_CHECK(cudaEventCreate(&ev_idx_stop));
            CUDA_CHECK(cudaEventCreate(&ev_dequant_stop));

            float idx_best = 1e9f, dequant_best = 1e9f, mm_best = 1e9f, total_best = 1e9f;
            for (int i = 0; i < iter; ++i) {
                CUDA_CHECK(cudaEventRecord(ev_start, stream));

                // idx (sparse activation computation)
                get_idx_cuda(d_idx, d_merge_idx, d_act_neurons, M, N, stream,
                             d_unique_idx, d_unique_neurons);
                CUDA_CHECK(cudaEventRecord(ev_idx_stop, stream));

                // sparse dequant (only unique active rows across all groups)
                dequantize_row_q4_0_sparse_unique_cuda(d_gate, d_gate_fp16, N, K,
                    d_unique_idx, d_unique_neurons, stream);
                dequantize_row_q4_0_sparse_unique_cuda(d_up, d_up_fp16, N, K,
                    d_unique_idx, d_unique_neurons, stream);
                dequantize_row_q4_0_sparse_unique_cuda(d_down, d_down_fp16, N, K,
                    d_unique_idx, d_unique_neurons, stream);
                CUDA_CHECK(cudaEventRecord(ev_dequant_stop, stream));

                // gate + up
                gate_and_up_mul_mat_cuda_sparse(
                    d_gate_fp16, d_up_fp16, d_input,
                    d_gate_out, d_up_out,
                    d_merge_idx, N, M, K, d_act_neurons,
                    stream, stream);

                // activation
                relu_and_mul_cuda(d_gate_out, d_up_out, d_gateup, M, N, stream);

                // down
                down_mul_mat_cuda_sparse(
                    d_down_fp16, d_gateup, d_output,
                    d_merge_idx, K, M, N, d_act_neurons, stream);

                CUDA_CHECK(cudaEventRecord(ev_stop, stream));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                CUDA_CHECK(cudaGetLastError());

                float t_idx, t_dequant, t_total;
                CUDA_CHECK(cudaEventElapsedTime(&t_idx,     ev_start, ev_idx_stop));
                CUDA_CHECK(cudaEventElapsedTime(&t_dequant, ev_idx_stop, ev_dequant_stop));
                CUDA_CHECK(cudaEventElapsedTime(&t_total,   ev_start, ev_stop));
                float t_mm = t_total - t_idx - t_dequant;

                idx_best     = fminf(idx_best,     t_idx);
                dequant_best = fminf(dequant_best, t_dequant);
                mm_best      = fminf(mm_best,      t_mm);

                printf("  SpBatch[%d]: idx=%f dequant=%f mm=%f total=%f ms\n", i, t_idx, t_dequant, t_mm, t_total);
            }
            total_best = idx_best + dequant_best + mm_best;
            total_spbatch_idx     += idx_best;
            total_spbatch_dequant += dequant_best;
            total_spbatch_mm      += mm_best;
            total_spbatch         += total_best;
            printf("  -> SpBatch best: idx=%f dequant=%f mm=%f total=%f ms\n",
                   idx_best, dequant_best, mm_best, total_best);

            CUDA_CHECK(cudaEventDestroy(ev_idx_stop));
            CUDA_CHECK(cudaEventDestroy(ev_dequant_stop));
        }

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
        printf("\n");
    }

    // ================================================================
    //  Results
    // ================================================================
    printf("\n");
    printf("====================================================================\n");
    printf("RESULTS (sum over %d layers, %d iters per layer, taking min)\n", LAYERS, iter);
    printf("====================================================================\n");
    printf("PowerInfer total:         %8.3f ms\n", total_powerinfer);
    printf("Cublas  dequant:          %8.3f ms\n", total_cublas_dequant);
    printf("Cublas  gemm:             %8.3f ms\n", total_cublas_gemm);
    printf("Cublas  total:            %8.3f ms\n", total_cublas_dequant + total_cublas_gemm);
    printf("SpBatch idx:              %8.3f ms\n", total_spbatch_idx);
    printf("SpBatch dequant:          %8.3f ms\n", total_spbatch_dequant);
    printf("SpBatch mm (sparse):      %8.3f ms\n", total_spbatch_mm);
    printf("SpBatch total:            %8.3f ms\n", total_spbatch);
    printf("--------------------------------------------------------------------\n");
    printf("CSV: %.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
           total_powerinfer,
           total_cublas_dequant, total_cublas_gemm,
           total_spbatch_idx, total_spbatch_dequant, total_spbatch_mm, total_spbatch);

    // Write CSV file (append mode)
    if (csv_name) {
        int write_header = 0;
        FILE *fp = fopen(csv_name, "r");
        if (!fp) write_header = 1;
        else fclose(fp);

        fp = fopen(csv_name, "a");
        if (fp) {
            if (write_header) {
                fprintf(fp, "M,N,K,iter,PowerInfer_total,"
                            "Cublas_dequant,Cublas_gemm,Cublas_total,"
                            "SpBatch_idx,SpBatch_dequant,SpBatch_mm,SpBatch_total\n");
            }
            fprintf(fp, "%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                    M, N, K, iter,
                    total_powerinfer,
                    total_cublas_dequant, total_cublas_gemm,
                    total_cublas_dequant + total_cublas_gemm,
                    total_spbatch_idx, total_spbatch_dequant, total_spbatch_mm,
                    total_spbatch);
            fclose(fp);
            printf("Results appended to %s\n", csv_name);
        } else {
            fprintf(stderr, "Warning: could not open %s for writing\n", csv_name);
        }
    }

    // cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_idx));
    CUDA_CHECK(cudaFree(d_gate));
    CUDA_CHECK(cudaFree(d_up));
    CUDA_CHECK(cudaFree(d_down));
    CUDA_CHECK(cudaFree(d_gate_fp16));
    CUDA_CHECK(cudaFree(d_up_fp16));
    CUDA_CHECK(cudaFree(d_down_fp16));
    CUDA_CHECK(cudaFree(d_gate_out));
    CUDA_CHECK(cudaFree(d_up_out));
    CUDA_CHECK(cudaFree(d_gate_act));
    CUDA_CHECK(cudaFree(d_gateup));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_merge_idx));
    CUDA_CHECK(cudaFree(d_act_neurons));
    CUDA_CHECK(cudaFree(d_unique_idx));
    CUDA_CHECK(cudaFree(d_unique_neurons));
    CUDA_CHECK(cudaFree(d_input_h));
    CUDA_CHECK(cudaFree(d_gateup_h));

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}
