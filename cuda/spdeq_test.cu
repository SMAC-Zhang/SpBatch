#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "kernel.cuh"

static void usage(const char* prog) {
    printf("Usage:\n");
    printf("  %s\n", prog);
    printf("  %s M N K active_percent [iter]\n", prog);
    printf("\n");
    printf("Defaults: M=8 N=11008 K=4096 active_percent=37 iter=20\n");
}

static bool validate_args(const int M, const int N, const int K, const int active_percent, const int iter) {
    if (M % SPARSITY_GROUP_SIZE != 0) {
        fprintf(stderr, "M must be a multiple of %d\n", SPARSITY_GROUP_SIZE);
        return false;
    }
    if (M / SPARSITY_GROUP_SIZE > MAX_GROUPS) {
        fprintf(stderr, "M / %d must be <= %d\n", SPARSITY_GROUP_SIZE, MAX_GROUPS);
        return false;
    }
    if (N % WARP_SIZE != 0) {
        fprintf(stderr, "N must be a multiple of %d for get_idx_cuda\n", WARP_SIZE);
        return false;
    }
    if (K % QK4_0 != 0) {
        fprintf(stderr, "K must be a multiple of %d for Q4_0\n", QK4_0);
        return false;
    }
    if (active_percent < 0 || active_percent > 100) {
        fprintf(stderr, "active_percent must be in [0, 100]\n");
        return false;
    }
    if (iter <= 0) {
        fprintf(stderr, "iter must be positive\n");
        return false;
    }
    return true;
}

static float rand_float(const float lo, const float hi) {
    const float t = (float)rand() / (float)RAND_MAX;
    return lo + (hi - lo) * t;
}

static void fill_random_q4_0(block_q4_0* data, const size_t blocks) {
    for (size_t i = 0; i < blocks; ++i) {
        data[i].d = __float2half(rand_float(0.001f, 2.0f));
        for (int j = 0; j < QK4_0 / 2; ++j) {
            data[i].qs[j] = (uint8_t)(rand() & 0xFF);
        }
    }
}

static void generate_idx(float* idx, const int M, const int N, const int active_percent) {
    std::fill(idx, idx + (size_t)M * N, -1.0f);
    const int groups = M / SPARSITY_GROUP_SIZE;

    for (int g = 0; g < groups; ++g) {
        int active_count = 0;
        for (int neuron = 0; neuron < N; ++neuron) {
            const bool active = ((neuron * 131 + g * 17) % 100) < active_percent;
            if (!active) {
                continue;
            }

            active_count++;
            const int token = g * SPARSITY_GROUP_SIZE + (neuron % SPARSITY_GROUP_SIZE);
            idx[(size_t)token * N + neuron] = 1.0f;
        }

        if (active_count == 0) {
            idx[(size_t)(g * SPARSITY_GROUP_SIZE) * N] = 1.0f;
        }
    }
}

struct CheckResult {
    size_t checked;
    size_t mismatches;
    float max_abs;
    int worst_neuron;
    int worst_col;
};

static CheckResult check_active_rows(const std::vector<half>& sparse,
                                     const std::vector<half>& dense,
                                     const std::vector<int>& merge_idx,
                                     const std::vector<int>& act_neurons,
                                     const int M, const int N, const int K) {
    const int groups = M / SPARSITY_GROUP_SIZE;
    CheckResult result{0, 0, 0.0f, -1, -1};

    for (int g = 0; g < groups; ++g) {
        const int prev = (g > 0) ? act_neurons[g - 1] : 0;
        const int actM = act_neurons[g] - prev;

        for (int r = 0; r < actM; ++r) {
            const int neuron = merge_idx[g * N + r];
            for (int col = 0; col < K; ++col) {
                const size_t pos = (size_t)neuron * K + col;
                const float actual = __half2float(sparse[pos]);
                const float expected = __half2float(dense[pos]);
                const float diff = fabsf(actual - expected);

                result.checked++;
                if (diff > result.max_abs) {
                    result.max_abs = diff;
                    result.worst_neuron = neuron;
                    result.worst_col = col;
                }
                if (diff != 0.0f) {
                    if (result.mismatches < 8) {
                        printf("mismatch group=%d active_row=%d neuron=%d col=%d sparse=%g dense=%g diff=%g\n",
                               g, r, neuron, col, actual, expected, diff);
                    }
                    result.mismatches++;
                }
            }
        }
    }

    return result;
}

template<typename Fn>
static float benchmark_ms(const char* name, const int iter, cudaStream_t stream, Fn fn) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < 3; ++i) {
        fn();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    float best = 1e30f;
    for (int i = 0; i < iter; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        fn();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        best = fminf(best, ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("%s best=%f ms\n", name, best);
    return best;
}

int main(int argc, char** argv) {
    srand(0);

    int M = 8;
    int N = 11008;
    int K = 4096;
    int active_percent = 37;
    int iter = 20;

    if (argc == 5 || argc == 6) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
        active_percent = atoi(argv[4]);
        if (argc == 6) {
            iter = atoi(argv[5]);
        }
    } else if (argc != 1) {
        usage(argv[0]);
        return 1;
    }

    if (!validate_args(M, N, K, active_percent, iter)) {
        return 1;
    }

    const int groups = M / SPARSITY_GROUP_SIZE;
    const size_t elems = (size_t)N * K;
    const size_t q4_blocks = elems / QK4_0;

    printf("spdeq_test: M=%d N=%d K=%d groups=%d active_percent=%d iter=%d\n",
           M, N, K, groups, active_percent, iter);

    std::vector<block_q4_0> h_q4(q4_blocks);
    std::vector<float> h_idx((size_t)M * N);
    fill_random_q4_0(h_q4.data(), q4_blocks);
    generate_idx(h_idx.data(), M, N, active_percent);

    block_q4_0* d_q4 = nullptr;
    float* d_idx = nullptr;
    half* d_dense = nullptr;
    half* d_sparse = nullptr;
    int* d_merge_idx = nullptr;
    int* d_act_neurons = nullptr;
    int* d_unique_idx = nullptr;
    int* d_unique_neurons = nullptr;

    CUDA_CHECK(cudaMalloc(&d_q4, q4_blocks * sizeof(block_q4_0)));
    CUDA_CHECK(cudaMalloc(&d_idx, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dense, elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_sparse, elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_merge_idx, (size_t)groups * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_act_neurons, groups * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_unique_idx, (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_unique_neurons, sizeof(int)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpyAsync(d_q4, h_q4.data(), q4_blocks * sizeof(block_q4_0),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_idx, h_idx.data(), (size_t)M * N * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_dense, 0, elems * sizeof(half), stream));
    CUDA_CHECK(cudaMemsetAsync(d_sparse, 0, elems * sizeof(half), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    get_idx_cuda(d_idx, d_merge_idx, d_act_neurons, M, N, stream, d_unique_idx, d_unique_neurons);
    dequantize_q4_0_full_cuda(d_q4, d_dense, (int)elems, stream);
    dequantize_row_q4_0_sparse_unique_cuda(d_q4, d_sparse, N, K, d_unique_idx, d_unique_neurons, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    std::vector<half> h_dense(elems);
    std::vector<half> h_sparse(elems);
    std::vector<int> h_merge_idx((size_t)groups * N);
    std::vector<int> h_act_neurons(groups);
    int h_unique_neurons = 0;

    CUDA_CHECK(cudaMemcpy(h_dense.data(), d_dense, elems * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sparse.data(), d_sparse, elems * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_merge_idx.data(), d_merge_idx, (size_t)groups * N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_act_neurons.data(), d_act_neurons, groups * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_unique_neurons, d_unique_neurons, sizeof(int), cudaMemcpyDeviceToHost));

    int prev = 0;
    for (int g = 0; g < groups; ++g) {
        const int actM = h_act_neurons[g] - prev;
        printf("group %d active_neurons=%d\n", g, actM);
        prev = h_act_neurons[g];
    }

    const int active_rows_total = groups > 0 ? h_act_neurons[groups - 1] : 0;
    std::vector<uint8_t> unique_active(N, 0);
    for (int g = 0; g < groups; ++g) {
        const int group_prev = (g > 0) ? h_act_neurons[g - 1] : 0;
        const int actM = h_act_neurons[g] - group_prev;
        for (int r = 0; r < actM; ++r) {
            unique_active[h_merge_idx[g * N + r]] = 1;
        }
    }
    int unique_active_rows = 0;
    for (int i = 0; i < N; ++i) {
        unique_active_rows += unique_active[i] ? 1 : 0;
    }
    if (h_unique_neurons != unique_active_rows) {
        fprintf(stderr, "unique count mismatch: get_idx=%d host_union=%d\n", h_unique_neurons, unique_active_rows);
        CUDA_CHECK(cudaFree(d_q4));
        CUDA_CHECK(cudaFree(d_idx));
        CUDA_CHECK(cudaFree(d_dense));
        CUDA_CHECK(cudaFree(d_sparse));
        CUDA_CHECK(cudaFree(d_merge_idx));
        CUDA_CHECK(cudaFree(d_act_neurons));
        CUDA_CHECK(cudaFree(d_unique_idx));
        CUDA_CHECK(cudaFree(d_unique_neurons));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 1;
    }

    const double sparse_work_ratio = (double)active_rows_total / (double)(groups * N);
    const double duplicate_rows_ratio = (double)active_rows_total / (double)N;
    const double unique_ratio = (double)h_unique_neurons / (double)N;
    printf("active rows total=%d / %d group_rows (%.2f%%), unique=%d / %d (%.2f%%)\n",
           active_rows_total, groups * N, sparse_work_ratio * 100.0,
           h_unique_neurons, N, unique_ratio * 100.0);

    const CheckResult result = check_active_rows(h_sparse, h_dense, h_merge_idx, h_act_neurons, M, N, K);
    printf("checked=%zu mismatches=%zu max_abs=%g", result.checked, result.mismatches, result.max_abs);
    if (result.worst_neuron >= 0) {
        printf(" worst=(neuron=%d,col=%d)", result.worst_neuron, result.worst_col);
    }
    printf("\n");

    if (result.mismatches != 0) {
        fprintf(stderr, "Sparse Q4_0 dequant check FAILED\n");
        CUDA_CHECK(cudaFree(d_q4));
        CUDA_CHECK(cudaFree(d_idx));
        CUDA_CHECK(cudaFree(d_dense));
        CUDA_CHECK(cudaFree(d_sparse));
        CUDA_CHECK(cudaFree(d_merge_idx));
        CUDA_CHECK(cudaFree(d_act_neurons));
        CUDA_CHECK(cudaFree(d_unique_idx));
        CUDA_CHECK(cudaFree(d_unique_neurons));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 1;
    }

    printf("Sparse Q4_0 dequant check PASSED\n");

    const float full_ms = benchmark_ms("full dequant", iter, stream, [&]() {
        dequantize_q4_0_full_cuda(d_q4, d_dense, (int)elems, stream);
    });
    const float duplicate_sparse_ms = benchmark_ms("duplicate sparse dequant", iter, stream, [&]() {
        dequantize_row_q4_0_sparse_cuda(d_q4, d_sparse, N, K, d_merge_idx, d_act_neurons, groups, stream);
    });
    const float unique_sparse_ms = benchmark_ms("unique sparse dequant", iter, stream, [&]() {
        dequantize_row_q4_0_sparse_unique_cuda(d_q4, d_sparse, N, K, d_unique_idx, d_unique_neurons, stream);
    });

    printf("perf summary: full=%f ms duplicate_sparse=%f ms unique_sparse=%f ms "
           "duplicate_speedup=%fx unique_speedup=%fx duplicate_rows=%.2f%% unique_rows=%.2f%%\n",
           full_ms, duplicate_sparse_ms, unique_sparse_ms,
           full_ms / duplicate_sparse_ms, full_ms / unique_sparse_ms,
           duplicate_rows_ratio * 100.0, unique_ratio * 100.0);
    if (unique_ratio < 0.80 && unique_sparse_ms >= full_ms) {
        fprintf(stderr, "Warning: unique sparse dequant was not faster despite unique_rows < 80%%; inspect kernel launch/work mapping.\n");
    }

    CUDA_CHECK(cudaFree(d_q4));
    CUDA_CHECK(cudaFree(d_idx));
    CUDA_CHECK(cudaFree(d_dense));
    CUDA_CHECK(cudaFree(d_sparse));
    CUDA_CHECK(cudaFree(d_merge_idx));
    CUDA_CHECK(cudaFree(d_act_neurons));
    CUDA_CHECK(cudaFree(d_unique_idx));
    CUDA_CHECK(cudaFree(d_unique_neurons));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}
