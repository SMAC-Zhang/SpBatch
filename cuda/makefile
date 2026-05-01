NVCC = nvcc
NVCCFLAGS = -arch=sm_89
CUBLASFLAGS = -lcublas

CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall

utility.o: utility.cu utility.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

kernel.o: kernel.cu kernel.cuh
	$(NVCC) $(NVCCFLAGS) -c  $< -o $@

gemm_test.o: gemm_test.cu utility.cuh kernel.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

gemm_test: utility.o kernel.o gemm_test.o
	$(NVCC) $(NVCCFLAGS) $(CUBLASFLAGS) $^ -o $@

sparsity_test.o: sparsity_test.cu utility.cuh kernel.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

sparsity_test: utility.o kernel.o sparsity_test.o
	$(NVCC) $(NVCCFLAGS) $(CUBLASFLAGS) $^ -o $@

spmm_test.o: spmm_test.cu utility.cuh kernel.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

spmm_test: utility.o kernel.o spmm_test.o
	$(NVCC) $(NVCCFLAGS) $(CUBLASFLAGS) $^ -o $@

ffn_test.o: ffn_test.cu utility.cuh kernel.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

ffn_test: utility.o kernel.o ffn_test.o
	$(NVCC) $(NVCCFLAGS) $(CUBLASFLAGS) $^ -o $@

quantize_test.o: quantize_test.cu utility.cuh kernel.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

quantize_test: utility.o kernel.o quantize_test.o
	$(NVCC) $(NVCCFLAGS) $(CUBLASFLAGS) $^ -o $@

dequantize_test.o: dequantize_test.cu utility.cuh kernel.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

dequantize_test: utility.o kernel.o dequantize_test.o
	$(NVCC) $(NVCCFLAGS) $(CUBLASFLAGS) $^ -o $@

spdeq_test.o: spdeq_test.cu kernel.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

spdeq_test: utility.o kernel.o spdeq_test.o
	$(NVCC) $(NVCCFLAGS) $(CUBLASFLAGS) $^ -o $@

all: gemm_test sparsity_test spmm_test ffn_test quantize_test dequantize_test spdeq_test

clean:
	rm -f *.o gemm_test sparsity_test spmm_test ffn_test quantize_test dequantize_test spdeq_test
