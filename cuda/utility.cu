#include <string>
#include <iostream>
#include <fstream>

#include "utility.cuh"

void read_idx(const int layer, const int hidden_size, const int tokens, void * buffer) {
    std::string filename = "../DejaVu_predictor/predictor_data/prosparse-llama-2-7b/mlp_label_" + std::to_string(layer) + ".mmap";

    const size_t bytes = tokens * hidden_size * sizeof(uint16_t);

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "can not open file: " << filename << std::endl;
        exit(1);
    }

    file.seekg(8 * sizeof(uint16_t) * hidden_size, std::ios::beg);

    if (!file.read(reinterpret_cast<char*>(buffer), bytes)) {
        std::cerr << "read file failed" << std::endl;
        exit(1);
    }

    return;
}

void randIdx(float * idx, size_t size, int p) {
    // int cnt = 0;
    for (size_t i = 0; i < size; i++) {
        idx[i] = -1.0f * abs(rand()) / (1.0f * (float)RAND_MAX);
        if (rand() % 100 >= p) {
            idx[i] = -idx[i];
        }
        // if (idx[i] > 0) {
        //     cnt++;
        // }
    }
    // printf("cnt: %d, size: %d\n, sp: %f", cnt, (int)size, 1.0f * cnt / size);
}

void generateIdx(float *idx, int M, int N, int p, int mp) {
    int k = (int)(N * p / 100.0 + 0.5);
    int mk = (int)(N * mp / 100.0 + 0.5);
    
    if (p < 0 || p > 100 || mp < 0 || mp > 100) {
        fprintf(stderr, "Error: p and mp must be between 0 and 100\n");
        return;
    }
    if (k < 0 || k > N || mk < 0 || mk > N) {
        fprintf(stderr, "Error: Invalid parameters for matrix dimensions\n");
        return;
    }
    if (mk > k || k - mk > N - mk - 1) {
        fprintf(stderr, "Error: Constraints not satisfied: mk <= k and k-mk <= N-mk-1\n");
        return;
    }
    
    srand(time(NULL));
    
    int *columns = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        columns[i] = i;
    }
    for (int i = N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = columns[i];
        columns[i] = columns[j];
        columns[j] = temp;
    }
    
    int *fullZeroCols = columns;
    int *remainingCols = columns + mk;
    int remainingCount = N - mk;
    
    for (int i = 0; i < M * N; i++) {
        idx[i] = 1.0f;
    }
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < mk; j++) {
            idx[i * N + fullZeroCols[j]] = 0.0f;
        }
    }
    
    if (k - mk > 0) {
        for (int i = 0; i < M; i++) {
            int *perm = (int*)malloc(remainingCount * sizeof(int));
            for (int j = 0; j < remainingCount; j++) {
                perm[j] = remainingCols[j];
            }
            for (int j = remainingCount - 1; j > 0; j--) {
                int r = rand() % (j + 1);
                int temp = perm[j];
                perm[j] = perm[r];
                perm[r] = temp;
            }
            
            for (int j = 0; j < k - mk; j++) {
                idx[i * N + perm[j]] = 0.0f;
            }
            free(perm);
        }
    }
    
    bool hasAllZero;
    do {
        hasAllZero = false;
        for (int j = 0; j < remainingCount; j++) {
            int col = remainingCols[j];
            bool allZero = true;
            
            for (int i = 0; i < M; i++) {
                if (idx[i * N + col] == 1.0f) {
                    allZero = false;
                    break;
                }
            }
            
            if (allZero) {
                hasAllZero = true;
                int row = rand() % M;
                
                idx[row * N + col] = 1.0f;
                
                int *candidates = (int*)malloc(remainingCount * sizeof(int));
                int count = 0;
                for (int jj = 0; jj < remainingCount; jj++) {
                    if (jj == j) continue;
                    int c = remainingCols[jj];
                    if (idx[row * N + c] == 1.0f) {
                        candidates[count++] = c;
                    }
                }
                
                if (count > 0) {
                    int r = rand() % count;
                    idx[row * N + candidates[r]] = 0.0f;
                } else {
                    fprintf(stderr, "Warning: No candidate column found for row %d\n", row);
                }
                free(candidates);
                break;
            }
        }
    } while (hasAllZero);
    
    free(columns);
}

static __global__ void compare_kernel(float * a, float * b, int M, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= M * N) return;

	int row = idx / N;
	int col = idx % N;

	if (std::abs(a[row * N + col] - b[row * N + col]) > 1.f) {
		printf("mismatch at (%d, %d): %f vs %f\n", row, col, a[row * N + col], b[row * N + col]);
	}
}

void compare_cuda(float * a, float * b, int M, int N, cudaStream_t stream) {
    compare_kernel<<<(M * N + 31) / 32, 32, 0, stream>>>(a, b, M, N);
}