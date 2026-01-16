#include <cstddef>
#include <cstdint>

#include "kernel.cuh"

/*************************** PTX instructions ***************************/
static __device__ __forceinline__ void CP_ASYNC_16(void* smem_ptr, const void* global_ptr, bool pred_guard = true) {
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], 16;\n"
                 "}\n" ::"r"((int)pred_guard),
                 "r"(smem_int_ptr),
                 "l"(global_ptr));
}

static __device__ __forceinline__ void RESET_ASYNC_16(void* smem_ptr, const void* global_ptr) {
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  cp.async.cg.shared.global [%0], [%1], 16, 0;\n"
                 "}\n" :: "r"(smem_int_ptr),
                 "l"(global_ptr));
}

static __device__ __forceinline__ void MMA_FP16_M16N8K16(uint32_t * __restrict__ a, uint32_t * __restrict__ b, uint32_t * __restrict__ c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                 "{ %0, %1, %2, %3 },"
                 "{ %4, %5, %6, %7 },"
                 "{ %8, %9 },"
                 "{ %10, %11, %12, %13 };"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]), 
                    "r"(b[0]),  "r"(b[1]), 
                    "r"(c[0]),  "r"(c[1]), "r"(c[2]),  "r"(c[3]));
}

template<class MMconfig>
static __device__ __forceinline__ void LD_UP_TILE_TO_SHARED(half* __restrict__ s_a, const half* __restrict__ a, const int LDA, 
                                                            half* __restrict__ s_b, const half* __restrict__ b, const int LDB) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int load_k_iter = MMconfig::BK / (SWIZZLE_UNIT * SWIZZLE_UNIT);
    
    { // load A_TILE
        int col = lane_id % SWIZZLE_UNIT;
        int row1 = lane_id / SWIZZLE_UNIT;
        int row2 = row1 + 4;
        int s_col1 = col ^ row1;
        int s_col2 = col ^ row2;

        const int copy_units = MMconfig::BM / SWIZZLE_UNIT;
        const int load_iter = (copy_units - 1)  / MMconfig::BLOCK_WARPS + 1;
        
        for (int i = 0; i < load_k_iter; ++i) {
            int unit_id = warp_id;
            for (int j = 0; j < load_iter; ++j) {
                bool pred = (unit_id < copy_units);
                half * __restrict__ s_local_a = s_a + unit_id * SWIZZLE_UNIT * MMconfig::BK;
                const half * __restrict__ local_a = a + unit_id * SWIZZLE_UNIT * LDA;
                CP_ASYNC_16(s_local_a + row1 * MMconfig::BK + s_col1 * SWIZZLE_UNIT, 
                            local_a + row1 * LDA + col * SWIZZLE_UNIT,        
                            pred);
                CP_ASYNC_16(s_local_a + row2 * MMconfig::BK + s_col2 * SWIZZLE_UNIT, 
                            local_a + row2 * LDA + col * SWIZZLE_UNIT,        
                            pred);
                unit_id += MMconfig::BLOCK_WARPS;
            }
            s_a += (SWIZZLE_UNIT * SWIZZLE_UNIT);
            a += (SWIZZLE_UNIT * SWIZZLE_UNIT);
        }
    }

    { // load B_TILE
        int row = lane_id % SWIZZLE_UNIT;
        int col1 = lane_id / SWIZZLE_UNIT;
        int col2 = col1 + 4;
        int s_row1 = row ^ col1;
        int s_row2 = row ^ col2;

        const int copy_units = MMconfig::BN / SWIZZLE_UNIT;
        const int load_iter = (copy_units - 1)  / MMconfig::BLOCK_WARPS + 1;
        
        for (int i = 0; i < load_k_iter; ++i) {
            int unit_id = warp_id;
            for (int j = 0; j < load_iter; ++j) {
                bool pred = (unit_id < copy_units);
                half * __restrict__ s_local_b = s_b + unit_id * SWIZZLE_UNIT * MMconfig::BK;
                const half * __restrict__ local_b = b + unit_id * SWIZZLE_UNIT * LDB;
                CP_ASYNC_16(s_local_b + col1 * MMconfig::BK + s_row1 * SWIZZLE_UNIT, 
                            local_b + col1 * LDB + row * SWIZZLE_UNIT,        
                            pred);
                CP_ASYNC_16(s_local_b + col2 * MMconfig::BK + s_row2 * SWIZZLE_UNIT, 
                            local_b + col2 * LDB + row * SWIZZLE_UNIT,        
                            pred);
                unit_id += MMconfig::BLOCK_WARPS;
            }
            s_b += (SWIZZLE_UNIT * SWIZZLE_UNIT);
            b += (SWIZZLE_UNIT * SWIZZLE_UNIT);
        }
    }
}

template<class MMconfig>
static __device__ __forceinline__ void LD_SPARSE_UP_TILE_TO_SHARED( half* __restrict__ s_a, const half* __restrict__ a, const int LDA, const int * merge_idx, int start_row, int actM,
                                                                    half* __restrict__ s_b, const half* __restrict__ b, const int LDB) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int load_k_iter = MMconfig::BK / (SWIZZLE_UNIT * SWIZZLE_UNIT);
    
    { // load A_TILE
        int col = lane_id % SWIZZLE_UNIT;
        int row1 = lane_id / SWIZZLE_UNIT;
        int row2 = row1 + 4;
        int s_col1 = col ^ row1;
        int s_col2 = col ^ row2;

        const int copy_units = MMconfig::BM / SWIZZLE_UNIT;
        const int load_iter = (copy_units - 1)  / MMconfig::BLOCK_WARPS + 1;
        
        #pragma unroll
        for (int i = 0; i < load_k_iter; ++i) {
            int unit_id = warp_id;
            #pragma unroll
            for (int j = 0; j < load_iter; ++j) {
                bool pred = (unit_id < copy_units);
                half * __restrict__ s_local_a = s_a + unit_id * SWIZZLE_UNIT * MMconfig::BK;
                const int a_row = unit_id * SWIZZLE_UNIT + start_row;
                pred = pred && (a_row + row1 < actM);
                CP_ASYNC_16(s_local_a + row1 * MMconfig::BK + s_col1 * SWIZZLE_UNIT, 
                            a + merge_idx[a_row + row1] * LDA + col * SWIZZLE_UNIT,        
                            pred);
                pred = pred && (a_row + row2 < actM);
                CP_ASYNC_16(s_local_a + row2 * MMconfig::BK + s_col2 * SWIZZLE_UNIT, 
                            a + merge_idx[a_row + row2] * LDA + col * SWIZZLE_UNIT,        
                            pred);
                unit_id += MMconfig::BLOCK_WARPS;
            }
            s_a += (SWIZZLE_UNIT * SWIZZLE_UNIT);
            a += (SWIZZLE_UNIT * SWIZZLE_UNIT);
        }
    }

    { // load B_TILE
        int row = lane_id % SWIZZLE_UNIT;
        int col1 = lane_id / SWIZZLE_UNIT;
        int col2 = col1 + 4;
        int s_row1 = row ^ col1;
        int s_row2 = row ^ col2;

        const int copy_units = MMconfig::BN / SWIZZLE_UNIT;
        const int load_iter = (copy_units - 1)  / MMconfig::BLOCK_WARPS + 1;
        
        #pragma unroll
        for (int i = 0; i < load_k_iter; ++i) {
            int unit_id = warp_id;
            #pragma unroll
            for (int j = 0; j < load_iter; ++j) {
                bool pred = (unit_id < copy_units);
                half * __restrict__ s_local_b = s_b + unit_id * SWIZZLE_UNIT * MMconfig::BK;
                const half * __restrict__ local_b = b + unit_id * SWIZZLE_UNIT * LDB;
                CP_ASYNC_16(s_local_b + col1 * MMconfig::BK + s_row1 * SWIZZLE_UNIT, 
                            local_b + col1 * LDB + row * SWIZZLE_UNIT,        
                            pred);
                CP_ASYNC_16(s_local_b + col2 * MMconfig::BK + s_row2 * SWIZZLE_UNIT, 
                            local_b + col2 * LDB + row * SWIZZLE_UNIT,        
                            pred);
                unit_id += MMconfig::BLOCK_WARPS;
            }
            s_b += (SWIZZLE_UNIT * SWIZZLE_UNIT);
            b += (SWIZZLE_UNIT * SWIZZLE_UNIT);
        }
    }
}


template<class MMconfig, int slda, int sldb>
static __device__ __forceinline__ void LD_SPARSE_DOWN_TILE_TO_SHARED(half* __restrict__ s_a, const half* __restrict__ a, const int LDA, const int * merge_idx, int start_k, int actK,
                                                                     half* __restrict__ s_b, const half* __restrict__ b, const int LDB) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    { // init A_TILE to 0
        int size_per_warp = (MMconfig::BM * MMconfig::BK) / MMconfig::BLOCK_WARPS;
        int iter = size_per_warp / WARP_SIZE / HALF_PER_FLOAT4;
        half* start_ptr = s_a + warp_id * size_per_warp + lane_id * HALF_PER_FLOAT4;
        #pragma unroll
        for (int i = 0; i < iter; ++i) {
            RESET_ASYNC_16(start_ptr, nullptr);
            start_ptr += WARP_SIZE * HALF_PER_FLOAT4;
        }
    }

    asm volatile("cp.async.wait_all;\n");

    { // load A_TILE
        int load_m_iter = MMconfig::BM / (SWIZZLE_UNIT * SWIZZLE_UNIT);
        int row = lane_id % SWIZZLE_UNIT;
        int col1 = lane_id / SWIZZLE_UNIT;
        int col2 = col1 + 4;
        int s_row1 = row ^ col1;
        int s_row2 = row ^ col2;

        const int copy_units = MMconfig::BK / SWIZZLE_UNIT;
        const int load_iter = (copy_units - 1)  / MMconfig::BLOCK_WARPS + 1;
        
        #pragma unroll
        for (int i = 0; i < load_m_iter; ++i) {
            int unit_id = warp_id;
            #pragma unroll
            for (int j = 0; j < load_iter; ++j) {
                bool pred = (unit_id < copy_units);
                half * __restrict__ s_local_a = s_a + unit_id * SWIZZLE_UNIT * slda;
                const int a_col = unit_id * SWIZZLE_UNIT + start_k;
                pred = pred && (a_col + col1 < actK);
                CP_ASYNC_16(s_local_a + col1 * slda + s_row1 * SWIZZLE_UNIT, 
                            a + merge_idx[a_col + col1] * LDA + row * SWIZZLE_UNIT,        
                            pred);
                pred = pred && (a_col + col2 < actK);
                CP_ASYNC_16(s_local_a + col2 * slda + s_row2 * SWIZZLE_UNIT, 
                            a + merge_idx[a_col + col2] * LDA + row * SWIZZLE_UNIT,        
                            pred);
                unit_id += MMconfig::BLOCK_WARPS;
            }
            s_a += (SWIZZLE_UNIT * SWIZZLE_UNIT);
            a += (SWIZZLE_UNIT * SWIZZLE_UNIT);
        }
    }

    { // load B_TILE
        int load_k_iter = MMconfig::BK / (SWIZZLE_UNIT * SWIZZLE_UNIT);
        int row = lane_id % SWIZZLE_UNIT;
        int col1 = lane_id / SWIZZLE_UNIT;
        int col2 = col1 + 4;
        int s_row1 = row ^ col1;
        int s_row2 = row ^ col2;

        const int copy_units = MMconfig::BN / SWIZZLE_UNIT;
        const int load_iter = (copy_units - 1)  / MMconfig::BLOCK_WARPS + 1;

        #pragma unroll
        for (int i = 0; i < load_k_iter; ++i) {
            int unit_id = warp_id;
            #pragma unroll
            for (int j = 0; j < load_iter; ++j) {
                bool pred = (unit_id < copy_units);
                half * __restrict__ s_local_b = s_b + unit_id * SWIZZLE_UNIT * sldb;
                const half * __restrict__ local_b = b + unit_id * SWIZZLE_UNIT * LDB;
                CP_ASYNC_16(s_local_b + col1 * sldb + s_row1 * SWIZZLE_UNIT, 
                            local_b + col1 * LDB + row * SWIZZLE_UNIT,        
                            pred);
                CP_ASYNC_16(s_local_b + col2 * sldb + s_row2 * SWIZZLE_UNIT, 
                            local_b + col2 * LDB + row * SWIZZLE_UNIT,        
                            pred);
                unit_id += MMconfig::BLOCK_WARPS;
            }
            s_b += (SWIZZLE_UNIT * SWIZZLE_UNIT);
            b += (SWIZZLE_UNIT * SWIZZLE_UNIT);
        }
    }
}

template<int NumOfTensors, int slda>
static __device__ __forceinline__ void LD_ROW_FRAG_X4(uint32_t (* __restrict__ reg)[4], const half* __restrict__ smem_ptr, const int warp_start_row, const int wk) {
    int lane_id = threadIdx.x % 32;
    int row = lane_id % MMA_M;
    int col = lane_id / MMA_M;
    
    smem_ptr += (warp_start_row + row) * slda + (wk + col * SWIZZLE_UNIT);
    uint32_t smem_local_ptr = __cvta_generic_to_shared(smem_ptr);
    smem_local_ptr = smem_local_ptr ^ ((row % SWIZZLE_UNIT) * SWIZZLE_UNIT * sizeof(half)); // eliminate bank conflict
    
    #pragma unroll
    for (int i = 0; i < NumOfTensors; ++i) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(reg[i][0]), "=r"(reg[i][1]), "=r"(reg[i][2]), "=r"(reg[i][3])
                        : "r"(smem_local_ptr));
        smem_local_ptr += slda * MMA_M * sizeof(half);
    }
}

template<int NumOfTensors, int slda>
static __device__ __forceinline__ void LD_COL_FRAG_X4(uint32_t (* __restrict__ reg)[4], const half* __restrict__ smem_ptr, const int warp_start_row, const int wk) {
    int lane_id = threadIdx.x % 32;
    int row = lane_id / 8 % 2;
    int col = lane_id / 16 * 8 + lane_id % 8;
    
    smem_ptr += warp_start_row + row * SWIZZLE_UNIT + (wk + col) * slda;
    uint32_t smem_local_ptr = __cvta_generic_to_shared(smem_ptr);
    // smem_local_ptr = smem_local_ptr ^ ((col % SWIZZLE_UNIT) * SWIZZLE_UNIT * sizeof(half)); // eliminate bank conflict
    uint32_t bank_conflict_offset = (col % SWIZZLE_UNIT) * SWIZZLE_UNIT * sizeof(half);
    
    #pragma unroll
    for (int i = 0; i < NumOfTensors; ++i) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(reg[i][0]), "=r"(reg[i][1]), "=r"(reg[i][2]), "=r"(reg[i][3])
                        : "r"(smem_local_ptr ^ bank_conflict_offset));
        smem_local_ptr += MMA_M * sizeof(half);
    }
}

template<int NumOfTensors, int slda>
static __device__ __forceinline__ void LD_COL_FRAG_X2(uint32_t (* __restrict__ reg) [2], const half* __restrict__ smem_ptr, const int warp_start_col, const int wk) {
    int lane_id = threadIdx.x % 32;
    int col = lane_id % 8;
    int row = lane_id / 8;
    
    smem_ptr += (warp_start_col + col) * slda + (wk + row * SWIZZLE_UNIT);
    uint32_t smem_local_ptr = __cvta_generic_to_shared(smem_ptr);
    smem_local_ptr = smem_local_ptr ^ ((col % SWIZZLE_UNIT) * SWIZZLE_UNIT * sizeof(half)); // eliminate bank conflict

    #pragma unroll
    for (int i = 0; i < NumOfTensors; ++i) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(reg[i][0]), "=r"(reg[i][1])
                        : "r"(smem_local_ptr));
        smem_local_ptr += slda * MMA_N * sizeof(half);
    }
}

/*************************** cuda kernels ***************************/
static __device__ void convert_f16(const void * vx, const int ib, const int iqs, dfloat2 & v) {
    const half * x = (const half *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x = x[ib + iqs + 0];
    v.y = x[ib + iqs + 1];
}

static __device__ void convert_f32(const void * vx, const int ib, const int iqs, dfloat2 & v) {
    const float * x = (const float *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x = x[ib + iqs + 0];
    v.y = x[ib + iqs + 1];
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int i = blockDim.x*blockIdx.x + 2*threadIdx.x;

    if (i >= k) {
        return;
    }

    const int ib = i/qk; // block index
    const int iqs = (i%qk)/qr; // quant index
    const int iybs = i - i%qk; // y block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0]        = v.x;
    y[iybs + iqs + y_offset] = v.y;
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
static __global__ void dequantize_mul_mat_batch_sparse(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows, int src1_cols, int dst_ne0,int * lst, float * idx) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int gpu_row = blockIdx.y*blockDim.y + threadIdx.y;

    if (gpu_row >= nrows) {
        return;
    }
    int row = lst ? lst[gpu_row] : gpu_row;


    const int tid = threadIdx.x;

    const int iter_stride = 2*GGML_CUDA_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
    const int y_offset = qr == 1 ? 1 : qk/2;
    float * loop_idx = idx;;
    dfloat * loop_y = (dfloat *)y;
    float * loop_dst = dst;



    float tmp = 0.0f;

    for (int col_id = 0; col_id < src1_cols; col_id++)
    {
        __syncthreads();
        tmp = 0.0f;
        if (loop_idx[row] <= 0.0f)
        {
            loop_dst += dst_ne0;
            loop_idx += dst_ne0;
            loop_y += ncols;
            continue;
        }

        for (int i = 0; i < ncols; i += iter_stride)
        {
            const int col = i + vals_per_iter * tid;
            const int ib = (gpu_row * ncols + col) / qk; // x block index
            const int iqs = (col % qk) / qr;         // x quant index
            const int iybs = col - col % qk;         // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
            for (int j = 0; j < vals_per_iter; j += 2)
            {
                // process 2 vals per j iter

                // dequantize
                // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
                dfloat2 v;
                dequantize_kernel(vx, ib, iqs + j / qr, v);

                // matrix multiplication

                tmp += v.x * loop_y[iybs + iqs + j / qr + 0];
                tmp += v.y * loop_y[iybs + iqs + j / qr + y_offset];
                // #endif
            }
        }
        atomicAdd(&loop_dst[row], tmp);
        loop_dst += dst_ne0;
        loop_idx += dst_ne0;
        loop_y += ncols;
    }
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
static __global__ void dequantize_mul_mat_axpy_sparse_batch(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows, int src1_ne0, int src1_ncols, int *lst, float *idx) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int gpu_row = blockIdx.y*blockDim.y + threadIdx.y;

    if (gpu_row >= nrows) {
        return;
    }
    int row = lst ? lst[gpu_row] : gpu_row;

    extern __shared__ float shared_dst[]; // TODO:dynamic

    const int tid = threadIdx.x;

    const int iter_stride = 2*GGML_CUDA_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
    const int y_offset = qr == 1 ? 1 : qk/2;
    float * loop_idx = idx;
    dfloat * loop_y = (dfloat *)y;
    float * loop_dst = dst;

// partial sum for each thread
    float tmp = 0.0f;
    for (int i = 0; i < ncols; i += GGML_CUDA_DMMV_X) {
        shared_dst[i+tid] = 0;
    }
    // __syncthreads();
    for (int col_id = 0; col_id < src1_ncols; col_id++) {
        __syncthreads();
        if (loop_idx[row] < dev_sparse_threshold) {
            loop_dst += ncols;
            loop_idx += src1_ne0;
            loop_y += src1_ne0;
            continue;
        }
        

        for (int i = 0; i < ncols; i += iter_stride)
        {
            const int col = i + vals_per_iter * tid;
            const int ib = (gpu_row * ncols + col) / qk; // x block index
            const int iqs = (col % qk) / qr;         // x quant index
            const int iybs = col - col % qk;         // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
            for (int j = 0; j < vals_per_iter; j += 2)
            {
                // process 2 vals per j iter

                // dequantize
                // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
                dfloat2 v;
                dequantize_kernel(vx, ib, iqs + j / qr, v);

                // matrix multiplication
                // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
                tmp = v.x * loop_y[row];
                shared_dst[iybs + iqs + j / qr + 0] = tmp;
                tmp = v.y * loop_y[row];
                shared_dst[iybs + iqs + j / qr + y_offset] = tmp;
            }
        }
        /* __syncthreads(); */

        for (int i = 0; i < ncols; i += GGML_CUDA_DMMV_X)
        {
            atomicAdd(&loop_dst[i + tid], shared_dst[i + tid]);
            shared_dst[i+tid] = 0;
        }
        loop_dst += ncols;
        loop_idx += src1_ne0;
        loop_y += src1_ne0;
    }
}

static __global__ void relu_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = fmaxf(x[i], 0);
}

static __global__ void mul_f32(const float * x, const float * y, float * dst, const int kx, const int ky) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= kx) {
        return;
    }
    dst[i] = x[i] * y[i%ky];
}

static __global__ void relu_and_mul_f32(const float * x, const float * y, float * dst, const int M, const int N) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= M * N) {
        return;
    }
    dst[i] = fmaxf(x[i], 0) * y[i];
}

static __global__ void relu_relu_mul_f32(const float * x, const float * y, float * dst, const int M, const int N) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= M * N) {
        return;
    }
    dst[i] = fmaxf(x[i], 0) * fmaxf(y[i], 0);
}

static __global__ void merge_batch_sparsity_kernel(const float * __restrict__ x, uint32_t * __restrict__ bits, uint32_t * __restrict__ prefix, const int batch_size, const int N, const int groups) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int wid = pos / WARP_SIZE;
    const int lane = pos % WARP_SIZE;
    const int len = N / WARP_SIZE;

    if (pos >= N) {
        return;
    }

    int pred[MAX_GROUPS] = {0};
    #pragma unroll
    for (int i = 0; i < batch_size; ++i) {
        pred[i / SPARSITY_GROUP_SIZE] += (x[i * N + pos] > 0);
    }

    #pragma unroll
    for (int i = 0; i < groups; ++i) {
        bits[i * len + wid] = __ballot_sync(0xFFFFFFFF, pred[i]);
    }
    if (lane < groups) {
        const int bits_pos = lane * len + wid;
        prefix[bits_pos] = __popc(bits[bits_pos]);
    }
}

// return: indics: [groups * neurons]
static __global__ void generate_indices_kernel(const uint32_t* __restrict__ bits, const uint32_t* __restrict__ prefix, int* __restrict__ indices, const int total_bits, const int groups) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int len = total_bits / groups;
    const int group_id = pos / len;
    const int neurons = len * WARP_SIZE;
    const int base_idx = pos * WARP_SIZE - group_id * neurons;

    if (pos >= total_bits) {
        return;
    }

    int output_pos = (pos == 0 ? 0 : prefix[pos - 1]) - (group_id == 0 ? 0 : prefix[group_id * len - 1]);

    uint32_t word = bits[pos];
    while (word) {
        int bit_pos = __ffs(word) - 1;
        
        indices[group_id * neurons + output_pos] = base_idx + bit_pos;
        output_pos++;
        word &= ~(1U << bit_pos);
    }
}

static __global__ void move_activation(const uint32_t* __restrict__ prefix, int* __restrict__ act_neurons_device, const int len) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    act_neurons_device[tid] = prefix[(tid + 1) * len - 1];
}

template<class MMConfig>
static __global__ void mm_up(const half * __restrict__ a, const half * __restrict__ b, float * __restrict__ c, const int M, const int N, const int K, const int * act_neurons, const int * merge_idx) {
    using namespace nvcuda;

    const int BM = MMConfig::BM;
    const int BN = MMConfig::BN;
    const int BK = MMConfig::BK;
    
    const int WARP_ROW_TENSORS = MMConfig::WARP_ROW_TENSORS;
    const int WARP_COL_TENSORS = MMConfig::WARP_COL_TENSORS;
    const int BLOCK_K_TENSORS = MMConfig::BLOCK_K_TENSORS;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int wrid = wid / MMConfig::COL_WARPS;
    const int wcid = wid % MMConfig::COL_WARPS;
    const int lane_id = tid % WARP_SIZE;

    const int LDA = K;
    const int LDB = K;
    const int LDC = M;
    const int actM = (bx == 0) ? act_neurons[0] : (act_neurons[bx] - act_neurons[bx - 1]);


    const int start_row = by * BM;
    const int start_col = bx * BN;
    if (start_row >= actM || start_col >= N) {
        return;
    }

    const int warp_start_row = wrid * WARP_ROW_TENSORS * MMA_M;
    const int warp_start_col = wcid * WARP_COL_TENSORS * MMA_N;

    // shared memory double buffer
    extern __shared__  __align__(128) char shared_mem[];
    half *s_a1 = (half *)shared_mem;
    half *s_a2 = (half *)&s_a1[BM * BK];
    half *s_b1 = (half *)&s_a2[BM * BK];
    half *s_b2 = (half *)&s_b1[BN * BK];
    merge_idx = merge_idx + bx * M;

    const int slda = BK;
    const int sldb = BK;

    uint32_t a_reg[2][WARP_ROW_TENSORS][4] = { 0 };
    uint32_t b_reg[2][WARP_COL_TENSORS][2] = { 0 };
    float c_reg[WARP_ROW_TENSORS][WARP_COL_TENSORS][4] = { 0.0f };

    #pragma unroll
    for (int sk = 0; ; sk += BK) {
        if (sk < K) {
            // load tile to shared memory
            LD_SPARSE_UP_TILE_TO_SHARED<MMConfig>
            (s_a2, a + sk, LDA, merge_idx, start_row, actM, s_b2, b + start_col * LDB + sk, LDB);
            asm volatile("cp.async.commit_group;\n" ::);
        }

        if (sk > 0) {
            // load frag from shared memory to registers
            LD_ROW_FRAG_X4<WARP_ROW_TENSORS, slda>(a_reg[0], s_a1, warp_start_row, 0);
            LD_COL_FRAG_X2<WARP_COL_TENSORS, sldb>(b_reg[0], s_b1, warp_start_col, 0);
            #pragma unroll
            for (int i = 0; i < BLOCK_K_TENSORS;) {
                #pragma unroll
                for (int j = 0; j < WARP_ROW_TENSORS; ++j) {
                    #pragma unroll
                    for (int k = 0; k < WARP_COL_TENSORS; ++k) {
                        MMA_FP16_M16N8K16(a_reg[i % 2][j], b_reg[i % 2][k], (uint32_t*)c_reg[j][k]);
                    }
                }
                i += 1;
                if (i >= BLOCK_K_TENSORS) {
                    break;
                }
                LD_ROW_FRAG_X4<WARP_ROW_TENSORS, slda>(a_reg[i % 2], s_a1, warp_start_row, i * MMA_K);
                LD_COL_FRAG_X2<WARP_COL_TENSORS, sldb>(b_reg[i % 2], s_b1, warp_start_col, i * MMA_K);
            }
        }

        if (sk >= K) {
            break;
        }
        
        // swap shared memory double buffer
        asm volatile("cp.async.wait_group 0;\n" ::);
        half* s_b = s_b1;
        s_b1 = s_b2;
        s_b2 = s_b;
        half* s_a = s_a1;
        s_a1 = s_a2;
        s_a2 = s_a;
        __syncthreads();
    }
    
    // transpose and write back C
    #pragma unroll
    for (int i = 0; i < WARP_ROW_TENSORS; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_COL_TENSORS; j++) {
            int row = warp_start_row + i * MMA_M + lane_id / 4 + start_row;
            int col = warp_start_col + j * MMA_N + lane_id % 4 * 2 + start_col;
            c[(col + 0) * LDC + row] = c_reg[i][j][0];
            c[(col + 0) * LDC + row + 8] = c_reg[i][j][2];
            c[(col + 1) * LDC + row] = c_reg[i][j][1];
            c[(col + 1) * LDC + row + 8] = c_reg[i][j][3];
        }
    }
}

template<class MMConfig>
static __global__ void mm_down(const half * __restrict__ a, const half * __restrict__ b, float * __restrict__ c, const int M, const int N, const int K, const int * act_neurons, const int * merge_idx) {
    using namespace nvcuda;

    const int BM = MMConfig::BM;
    const int BN = MMConfig::BN;
    const int BK = MMConfig::BK;
    
    const int WARP_ROW_TENSORS = MMConfig::WARP_ROW_TENSORS;
    const int WARP_COL_TENSORS = MMConfig::WARP_COL_TENSORS;
    const int BLOCK_K_TENSORS = MMConfig::BLOCK_K_TENSORS;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int wrid = wid / MMConfig::COL_WARPS;
    const int wcid = wid % MMConfig::COL_WARPS;
    const int lane_id = tid % WARP_SIZE;

    const int LDA = M;
    const int LDB = K;
    const int LDC = M;
    const int actK = (bx == 0) ? act_neurons[0] : (act_neurons[bx] - act_neurons[bx - 1]);


    const int start_row = by * BM;
    const int start_col = bx * BN;
    if (start_row >= M || start_col >= N) {
        return;
    }

    const int warp_start_row = wrid * WARP_ROW_TENSORS * MMA_M;
    const int warp_start_col = wcid * WARP_COL_TENSORS * MMA_N;

    // shared memory double buffer
    extern __shared__  __align__(128) char shared_mem[];
    half *s_a1 = (half *)shared_mem;
    half *s_a2 = (half *)&s_a1[BM * BK];
    half *s_b1 = (half *)&s_a2[BM * BK];
    half *s_b2 = (half *)&s_b1[BN * BK];
    merge_idx = merge_idx + bx * K;

    const int slda = BM;
    const int sldb = BK;

    uint32_t a_reg[2][WARP_ROW_TENSORS][4] = { 0 };
    uint32_t b_reg[2][WARP_COL_TENSORS][2] = { 0 };
    float c_reg[WARP_ROW_TENSORS][WARP_COL_TENSORS][4] = {0.0f};

    #pragma unroll
    for (int sk = 0; ; sk += BK) {
        if (sk < actK) {
            // load tile to shared memory
            LD_SPARSE_DOWN_TILE_TO_SHARED<MMConfig, slda, sldb>
            (s_a2, a + start_row, LDA, merge_idx, sk, actK, s_b2, b + start_col * LDB + sk, LDB);
            asm volatile("cp.async.commit_group;\n" ::);
        }

        if (sk > 0) {
            // load frag from shared memory to registers
            LD_COL_FRAG_X4<WARP_ROW_TENSORS, slda>(a_reg[0], s_a1, warp_start_row, 0);
            LD_COL_FRAG_X2<WARP_COL_TENSORS, sldb>(b_reg[0], s_b1, warp_start_col, 0);
            #pragma unroll
            for (int i = 0; i < BLOCK_K_TENSORS;) {
                #pragma unroll
                for (int j = 0; j < WARP_ROW_TENSORS; ++j) {
                    #pragma unroll
                    for (int k = 0; k < WARP_COL_TENSORS; ++k) {
                        MMA_FP16_M16N8K16(a_reg[i % 2][j], b_reg[i % 2][k], (uint32_t*)c_reg[j][k]);
                    }
                }
                i += 1;
                if (i >= BLOCK_K_TENSORS) {
                    break;
                }
                LD_COL_FRAG_X4<WARP_ROW_TENSORS, slda>(a_reg[i % 2], s_a1, warp_start_row, i * MMA_K);
                LD_COL_FRAG_X2<WARP_COL_TENSORS, sldb>(b_reg[i % 2], s_b1, warp_start_col, i * MMA_K);
            }
        }

        if (sk >= actK) {
            break;
        }
        
        // swap shared memory double buffer
        asm volatile("cp.async.wait_group 0;\n" ::);
        half* s_b = s_b1;
        s_b1 = s_b2;
        s_b2 = s_b;
        half* s_a = s_a1;
        s_a1 = s_a2;
        s_a2 = s_a;
        __syncthreads();
    }
    
    // transpose and write back C
    #pragma unroll
    for (int i = 0; i < WARP_ROW_TENSORS; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_COL_TENSORS; j++) {
            int row = warp_start_row + i * MMA_M + lane_id / 4 + start_row;
            int col = warp_start_col + j * MMA_N + lane_id % 4 * 2 + start_col;
            // printf("row: %d, col: %d\n", row, col);
            c[(col + 0) * LDC + row] = c_reg[i][j][0];
            c[(col + 0) * LDC + row + 8] = c_reg[i][j][2];
            c[(col + 1) * LDC + row] = c_reg[i][j][1];
            c[(col + 1) * LDC + row + 8] = c_reg[i][j][3];
        }
    }
}

static __global__ void add_f16_f32_f16_sparse(const float* __restrict__ a, const half * __restrict__ b, float * __restrict__ c, const int M, const int N, const int * act_neurons, const int * merge_idx) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int group_id = col / SPARSITY_GROUP_SIZE;

    int actM = (group_id > 0) ? (act_neurons[group_id] - act_neurons[group_id - 1]) : act_neurons[group_id];
    merge_idx = merge_idx + group_id * M;

    if (row >= actM || col >= N) {
        return;
    }
    c[col * M + row] = a[col * M + row] + __half2float(b[merge_idx[row]]);
}

/******************************** wrapper functions for CUDA kernels *****************************/
void convert_fp32_to_fp16_cuda(const void * vx, half * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    dequantize_block<1, 1, convert_f32><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

void convert_fp16_to_fp32_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<1, 1, convert_f16><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

void convert_mul_mat_batch_f16_cuda_sparse(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, int src1_ncols, int dst_ne0, cudaStream_t stream, int *lst, float *idx) {

    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    dequantize_mul_mat_batch_sparse<1, 1, convert_f16>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows, src1_ncols, dst_ne0, lst, idx);
}

void convert_axpy_sparse_batch_f16_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, int src1_rows, int src1_ncols, cudaStream_t stream, int *lst, float *idx) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    dequantize_mul_mat_axpy_sparse_batch<1, 1, convert_f16>
        <<<block_nums, block_dims, ncols*sizeof(float), stream>>>(vx, y, dst, ncols, nrows, src1_rows, src1_ncols, lst, idx);
}

void relu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    relu_f32<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

void mul_f32_cuda(const float * x, const float * y, float * dst, const int kx, const int ky, cudaStream_t stream) {
    const int num_blocks = (kx + CUDA_MUL_BLOCK_SIZE - 1) / CUDA_MUL_BLOCK_SIZE;
    mul_f32<<<num_blocks, CUDA_MUL_BLOCK_SIZE, 0, stream>>>(x, y, dst, kx, ky);
}


// parameters:
// M: batch size
// N: neurons
// return:
// merge_idx: groups * neurons * sizeof(int)
// act_neurons: number of active neurons
void get_idx_cuda(const float * idx, int * merge_idx, int * act_neurons_device, const int M, const int N, cudaStream_t stream0) {
    const int groups = M / SPARSITY_GROUP_SIZE;
    const int len = N / WARP_SIZE;
    const int total_bits = len * groups;

    static uint32_t *bits = nullptr, *prefix = nullptr;
    if (bits == nullptr) {
        CUDA_CHECK(cudaMalloc(&bits, sizeof(uint32_t) * N / WARP_SIZE * groups));
        CUDA_CHECK(cudaMalloc(&prefix, sizeof(uint32_t) * N / WARP_SIZE * groups));
    }

    {
        const int block_size = 4 * WARP_SIZE;
        const int block_num_x = (N + block_size - 1) / block_size;
        // printf("1、block_num_x: %d\n", block_num_x);

        const dim3 block_nums(block_num_x, 1, 1);
        const dim3 block_dims(block_size, 1, 1);
        merge_batch_sparsity_kernel<<<block_nums, block_dims, 0, stream0>>>(idx, bits, prefix, M, N, groups);
    }

    // prefix sum
    static void *d_temp_storage = nullptr;
    static size_t temp_storage_bytes = 0;
    if (d_temp_storage == nullptr) {
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, prefix, prefix, total_bits, stream0);
        CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    }
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, prefix, prefix, total_bits, stream0);

    // int act_neurons[groups];
    // CUDA_CHECK(cudaStreamSynchronize(stream0));

    move_activation<<<1, groups, 0, stream0>>>(prefix, act_neurons_device, len);
    // for (int i = 0; i < groups; ++i) {
    //     CUDA_CHECK(cudaMemcpyAsync(&act_neurons_device[i], &prefix[(i + 1) * len - 1], sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream0));
    // }
    // CUDA_CHECK(cudaMemcpyAsync(act_neurons, act_neurons_device, groups * sizeof(int), cudaMemcpyDeviceToHost, stream1));

    {
        const int block_size = 1 * WARP_SIZE;
        const int block_num_x = (total_bits + block_size - 1) / block_size;
        // printf("2、block_num_x: %d\n", block_num_x);

        const dim3 block_nums(block_num_x, 1, 1);
        const dim3 block_dims(block_size, 1, 1);
        generate_indices_kernel<<<block_nums, block_dims, 0, stream0>>>(bits, prefix, merge_idx, total_bits, groups);
    }

    // CUDA_CHECK(cudaStreamSynchronize(stream1));
    // act_N = 0;
    // #pragma unroll
    // for (int i = 0; i < groups; ++i) {
    //     if (i > 0) {
    //         act_N = std::max(act_N, act_neurons[i] - act_neurons[i - 1]);
    //     } else {
    //         act_N = std::max(act_N, act_neurons[i]);
    //     }
    // }
    CUDA_CHECK(cudaStreamSynchronize(stream0));
}

void up_mul_mat_cuda_sparse(const half* weight, const float* input, float* dst, int * merge_idx, const int M, const int N, const int K, const int * act_neurons_device, cudaStream_t stream) {
    static half* input_h = nullptr;
    
    if (input_h == nullptr) {
        CUDA_CHECK(cudaMalloc(&input_h, N * K * sizeof(half)));
    }

    convert_fp32_to_fp16_cuda(input, input_h, N * K, stream);

    using MMconfig = SparseMMConfig<4, 1, 4, 1, 4>;
    const int BN = MMconfig::BN;
    const int BM = MMconfig::BM;
    const int BK = MMconfig::BK;
    const int BLOCK_WARPS = MMconfig::BLOCK_WARPS;

    const int block_num_x = (N + BN - 1) / BN;
    const int block_num_y = (M + BM - 1) / BM;

    // printf("block_num_x: %d, block_num_y: %d, act_M: %d, batch_size: %d\n", block_num_x, block_num_y, act_M, N);
    // printf("BN: %d, BM: %d, BK: %d, BLOCK_WARPS: %d\n", BN, BM, BK, BLOCK_WARPS);
    
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(BLOCK_WARPS * WARP_SIZE, 1, 1);
    
    size_t shared_size = 2 * (BM * BK + BN * BK) * sizeof(half);

    cudaFuncSetAttribute(mm_up<MMconfig>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size);

    // printf("shared_size: %zu KB\n", shared_size / 1024);
    mm_up<MMconfig><<<block_nums, block_dims, shared_size, stream>>>(weight, input_h, dst, M, N, K, act_neurons_device, merge_idx);
}

void gate_and_up_mul_mat_cuda_sparse(const half* gate_w, const half* up_w, const float* input, float* gate_dst, float* up_dst, int * merge_idx, const int M, const int N, const int K, const int * act_neurons_device, cudaStream_t stream0, cudaStream_t stream1) {
    static half* input_h = nullptr;
    
    if (input_h == nullptr) {
        CUDA_CHECK(cudaMalloc(&input_h, N * K * sizeof(half)));
    }

    convert_fp32_to_fp16_cuda(input, input_h, N * K, stream0);

    using MMconfig = SparseMMConfig<4, 1, 4, 1, 4>;
    const int BN = MMconfig::BN;
    const int BM = MMconfig::BM;
    const int BK = MMconfig::BK;
    const int BLOCK_WARPS = MMconfig::BLOCK_WARPS;

    const int block_num_x = (N + BN - 1) / BN;
    const int block_num_y = (M + BM - 1) / BM;

    // printf("block_num_x: %d, block_num_y: %d, act_M: %d, batch_size: %d\n", block_num_x, block_num_y, act_M, N);
    // printf("BN: %d, BM: %d, BK: %d, BLOCK_WARPS: %d\n", BN, BM, BK, BLOCK_WARPS);
    
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(BLOCK_WARPS * WARP_SIZE, 1, 1);
    
    size_t shared_size = 2 * (BM * BK + BN * BK) * sizeof(half);

    cudaFuncSetAttribute(mm_up<MMconfig>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size);

    // printf("shared_size: %zu KB\n", shared_size / 1024);

    // gate
    mm_up<MMconfig><<<block_nums, block_dims, shared_size, stream0>>> \
    (gate_w, input_h, gate_dst, M, N, K, act_neurons_device, merge_idx);
    // up
    mm_up<MMconfig><<<block_nums, block_dims, shared_size, stream1>>> \
    (up_w, input_h, up_dst, M, N, K, act_neurons_device, merge_idx);

    // CUDA_CHECK(cudaStreamSynchronize(stream0));
    CUDA_CHECK(cudaStreamSynchronize(stream1));
}

void down_mul_mat_cuda_sparse(const half* weight, const float* input, float* dst, int * merge_idx, const int M, const int N, const int K, const int * act_neurons_device, cudaStream_t stream) {
    static half* input_h = nullptr;
    
    if (input_h == nullptr) {
        CUDA_CHECK(cudaMalloc(&input_h, N * K * sizeof(half)));
    }

    convert_fp32_to_fp16_cuda(input, input_h, N * K, stream);

    using MMconfig = SparseMMConfig<4, 1, 4, 1, 4>;
    const int BN = MMconfig::BN;
    const int BM = MMconfig::BM;
    const int BK = MMconfig::BK;
    const int BLOCK_WARPS = MMconfig::BLOCK_WARPS;

    const int block_num_x = (N + BN - 1) / BN;
    const int block_num_y = (M + BM - 1) / BM;

    // printf("block_num_x: %d, block_num_y: %d, act_K: %d, batch_size: %d\n", block_num_x, block_num_y, act_K, N);
    // printf("BN: %d, BM: %d, BK: %d, BLOCK_WARPS: %d\n", BN, BM, BK, BLOCK_WARPS);
    
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(BLOCK_WARPS * WARP_SIZE, 1, 1);
    
    size_t shared_size = 2 * (BM * BK + BK * BN) * sizeof(half);

    cudaFuncSetAttribute(mm_down<MMconfig>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size);

    // printf("shared_size: %zu KB\n", shared_size / 1024);
    mm_down<MMconfig><<<block_nums, block_dims, shared_size, stream>>>(weight, input_h, dst, M, N, K, act_neurons_device, merge_idx);
    // CUDA_CHECK(cudaStreamSynchronize(stream));
}

void relu_and_mul_cuda(const float * gate, const float* up, float * dst, const int M, const int N, cudaStream_t stream) {
    const int num_blocks = ((M * N) + CUDA_RELU_MUL_BLOCK_SIZE - 1) / CUDA_RELU_MUL_BLOCK_SIZE;
    relu_and_mul_f32<<<num_blocks, CUDA_RELU_MUL_BLOCK_SIZE, 0, stream>>>(gate, up, dst, M, N);
}

void relu_relu_mul_cuda(const float * gate, const float* up, float * dst, const int M, const int N, cudaStream_t stream) {
    const int num_blocks = ((M * N) + CUDA_RELU_MUL_BLOCK_SIZE - 1) / CUDA_RELU_MUL_BLOCK_SIZE;
    relu_relu_mul_f32<<<num_blocks, CUDA_RELU_MUL_BLOCK_SIZE, 0, stream>>>(gate, up, dst, M, N);
}

void add_sparse_cuda(const float* a, const half* b, float* c, const int M, const int N, const int * act_neurons_device, const int * merge_idx, cudaStream_t stream) {
    const int block_num_x = (N + SPARSITY_GROUP_SIZE - 1) / SPARSITY_GROUP_SIZE;
    const int block_num_y = (M + GGML_CUDA_DMMV_X - 1) / GGML_CUDA_DMMV_X;

    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(SPARSITY_GROUP_SIZE, GGML_CUDA_DMMV_X, 1);

    add_f16_f32_f16_sparse<<<block_nums, block_dims, 0, stream>>>(a, b, c, M, N, act_neurons_device, merge_idx);
}

/*******************test gemm *********************/
static __global__ void naive_gemm_kernel(const half * __restrict__ a, const half * __restrict__ b, float * __restrict__ c, const int M, const int N, const int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += __half2float(a[row * K + k]) * __half2float(b[col * K + k]);
    }
    c[row * N + col] = sum;
}

static __global__ void naive_mm_up_kernel(const half * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const int M, const int N, const int K, const int * act_neurons, int * merge_idx) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int group_id = col / SPARSITY_GROUP_SIZE;

    int actM = (group_id > 0) ? (act_neurons[group_id] - act_neurons[group_id - 1]) : act_neurons[group_id];
    merge_idx = merge_idx + group_id * M;

    if (row >= actM || col >= N) {
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += __half2float(a[merge_idx[row] * K + k]) * b[col * K + k];
    }
    c[row + col * M] = sum;
}

static __global__ void naive_mm_down_kernel(const half * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const int M, const int N, const int K, const int * act_neurons, int * merge_idx) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int group_id = col / SPARSITY_GROUP_SIZE;

    int actK = (group_id > 0) ? (act_neurons[group_id] - act_neurons[group_id - 1]) : act_neurons[group_id];
    merge_idx = merge_idx + group_id * K;

    if (row >= M || col >= N) {
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < actK; ++k) {
        sum += __half2float(a[row + merge_idx[k] * M]) * b[col * K + k];
    }
    c[row + col * M] = sum;
}

template<class MMConfig>
static __global__ void gemm_kernel(const half * __restrict__ a, const half * __restrict__ b, float * __restrict__ c, const int M, const int N, const int K) {
    using namespace nvcuda;

    const int BM = MMConfig::BM;
    const int BN = MMConfig::BN;
    const int BK = MMConfig::BK;
    
    const int WARP_ROW_TENSORS = MMConfig::WARP_ROW_TENSORS;
    const int WARP_COL_TENSORS = MMConfig::WARP_COL_TENSORS;
    const int BLOCK_K_TENSORS = MMConfig::BLOCK_K_TENSORS;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int wrid = wid / MMConfig::COL_WARPS;
    const int wcid = wid % MMConfig::COL_WARPS;
    const int lane_id = tid % WARP_SIZE;

    const int LDA = K;
    const int LDB = K;
    const int LDC = N;

    const int start_row = by * BM;
    const int start_col = bx * BN;
    if (start_row >= M || start_col >= N) {
        return;
    }

    const int warp_start_row = wrid * WARP_ROW_TENSORS * MMA_M;
    const int warp_start_col = wcid * WARP_COL_TENSORS * MMA_N;

    extern __shared__  __align__(128) char shared_mem[];
    half *s_a1 = (half *)shared_mem;
    half *s_a2 = (half *)&s_a1[BM * BK];
    half *s_b1 = (half *)&s_a2[BM * BK];
    half *s_b2 = (half *)&s_b1[BN * BK];

    const int slda = BK;
    const int sldb = BK;

    uint32_t a_reg[2][WARP_ROW_TENSORS][4];
    uint32_t b_reg[2][WARP_COL_TENSORS][2];
    float c_reg[WARP_ROW_TENSORS][WARP_COL_TENSORS][4] = {0.0f};

    #pragma unroll
    for (int sk = 0; ; sk += BK) {
        if (sk < K) {
            // load tile to shared memory
            LD_UP_TILE_TO_SHARED<MMConfig>
            (s_a2, a + start_row * LDA + sk, LDA, s_b2, b + start_col * LDB + sk, LDB);
            asm volatile("cp.async.commit_group;\n" ::);
        }

        if (sk > 0) {
            // load frag from shared memory to registers
            LD_ROW_FRAG_X4<WARP_ROW_TENSORS, slda>(a_reg[0], s_a1, warp_start_row, 0);
            LD_COL_FRAG_X2<WARP_COL_TENSORS, sldb>(b_reg[0], s_b1, warp_start_col, 0);
            #pragma unroll
            for (int i = 0; i < BLOCK_K_TENSORS;) {
                #pragma unroll
                for (int j = 0; j < WARP_ROW_TENSORS; ++j) {
                    #pragma unroll
                    for (int k = 0; k < WARP_COL_TENSORS; ++k) {
                        MMA_FP16_M16N8K16(a_reg[i % 2][j], b_reg[i % 2][k], (uint32_t*)c_reg[j][k]);
                    }
                }
                i += 1;
                if (i >= BLOCK_K_TENSORS) {
                    break;
                }
                LD_ROW_FRAG_X4<WARP_ROW_TENSORS, slda>(a_reg[i % 2], s_a1, warp_start_row, i * MMA_K);
                LD_COL_FRAG_X2<WARP_COL_TENSORS, sldb>(b_reg[i % 2], s_b1, warp_start_col, i * MMA_K);
            }
        }

        if (sk >= K) {
            break;
        }
        
        asm volatile("cp.async.wait_group 0;\n" ::);
        half* s_b = s_b1;
        s_b1 = s_b2;
        s_b2 = s_b;
        half* s_a = s_a1;
        s_a1 = s_a2;
        s_a2 = s_a;
        __syncthreads();
    }
    
    #pragma unroll
    for (int j = 0; j < WARP_COL_TENSORS; j++) {
        #pragma unroll
        for (int i = 0; i < WARP_ROW_TENSORS; i++) {
            int row = warp_start_row + i * MMA_M + lane_id / 4 + start_row;
            int col = warp_start_col + j * MMA_N + lane_id % 4 * 2 + start_col;
            c[row * LDC + col + 0] = c_reg[i][j][0];
            c[row * LDC + col + 1] = c_reg[i][j][1];
            row += 8;
            c[row * LDC + col + 0] = c_reg[i][j][2];
            c[row * LDC + col + 1] = c_reg[i][j][3];
        }
    }
}

void gemm_cuda(const half* a, const float* b, float* c, const int M, const int N, const int K, cudaStream_t stream) {
	static half * b_h = nullptr;
    if (b_h == nullptr) {
        CUDA_CHECK(cudaMalloc(&b_h, N * K * sizeof(half)));
    }

    convert_fp32_to_fp16_cuda(b, b_h, N * K, stream);
	using MMconfig = SparseMMConfig<2, 2, 4, 4, 4>;
    const int BN = MMconfig::BN;
    const int BM = MMconfig::BM;
    const int BK = MMconfig::BK;
    const int BLOCK_WARPS = MMconfig::BLOCK_WARPS;

    const int block_num_x = (N + BN - 1) / BN;
    const int block_num_y = (M + BM - 1) / BM;

    printf("block_num_x: %d, block_num_y: %d\n", block_num_x, block_num_y);
    printf("BM: %d, BN: %d, BK: %d, BLOCK_WARPS: %d\n", BM, BN, BK, BLOCK_WARPS);
    
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(BLOCK_WARPS * WARP_SIZE, 1, 1);
    
    size_t shared_size = 2 * (BM * BK + BN * BK) * sizeof(half);
	printf("shared_size: %zu KB\n", shared_size / 1024);

    cudaFuncSetAttribute(gemm_kernel<MMconfig>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size);
    gemm_kernel<MMconfig><<<block_nums, block_dims, shared_size, stream>>>(a, b_h, c, M, N, K);
}

void naive_gemm_cuda(const half* a, const half* b, float* c, const int M, const int N, const int K, cudaStream_t stream) {
    const int block_num_x = (N + 8 - 1) / 8;
    const int block_num_y = (M + 32 - 1) / 32;

    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(8, 32, 1);
    
    naive_gemm_kernel<<<block_nums, block_dims, 0, stream>>>(a, b, c, M, N, K);
}

void naive_mm_up_cuda(const half* a, const float* b, float* c, const int M, const int N, const int K, const int * act_neurons_device, int * merge_idx, cudaStream_t stream) {
    const int block_num_x = (N + 8 - 1) / 8;
    const int block_num_y = (M + 32 - 1) / 32;

    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(8, 32, 1);
    
    naive_mm_up_kernel<<<block_nums, block_dims, 0, stream>>>(a, b, c, M, N, K, act_neurons_device, merge_idx);
}

void naive_mm_down_cuda(const half* a, const float* b, float* c, const int M, const int N, const int K, const int * act_neurons_device, int * merge_idx, cudaStream_t stream) {
    const int block_num_x = (N + 8 - 1) / 8;
    const int block_num_y = (M + 32 - 1) / 32;

    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(8, 32, 1);
    
    naive_mm_down_kernel<<<block_nums, block_dims, 0, stream>>>(a, b, c, M, N, K, act_neurons_device, merge_idx);
}

void naive_sparsity_div(const float* idx, int * merge_idx, int* act_neurons, int M, int N) {
    const int groups = M / SPARSITY_GROUP_SIZE;

    for (int i = 0; i < groups; ++i) {
        int cnt = 0;
        for (int j = 0; j < N; ++j) {
            int pred = 0;
            for (int k = 0; k < SPARSITY_GROUP_SIZE; ++k) {
                pred += (idx[(i * SPARSITY_GROUP_SIZE + k) * N + j] > 0.0f);
            }
            if (pred > 0) {
                merge_idx[i * N + cnt] = j;
                cnt++;
            }
        }
        act_neurons[i] = cnt;
    }
}