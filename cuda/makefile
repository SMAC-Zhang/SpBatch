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

all: gemm_test sparsity_test spmm_test ffn_test

clean:
	rm -f *.o gemm_test sparsity_test spmm_test ffn_test