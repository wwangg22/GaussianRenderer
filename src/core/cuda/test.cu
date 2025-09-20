#include <cuda_runtime.h>
#include <iostream>
#include <cmath> 

#define TILEWIDTH 32

#define X_MULT 2

__global__ void matrixMul(float* A, float* B, float*result,  int N) {

    __shared__ float s_A[TILEWIDTH][TILEWIDTH];
    __shared__ float s_B[TILEWIDTH][TILEWIDTH * X_MULT];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x * X_MULT + threadIdx.x;
    float preload_a = A[row*N + threadIdx.x];
    float preload_b = B[threadIdx.y*N + col];

    float preload_b_2 = B[threadIdx.y*N + col + TILEWIDTH];
    float preload_b_3 = B[threadIdx.y*N + col + 2*TILEWIDTH];
    float preload_b_4 = B[threadIdx.y*N + col + 3*TILEWIDTH];
    float preload_b_5 = B[threadIdx.y*N + col + 4*TILEWIDTH];
    float preload_b_6 = B[threadIdx.y*N + col + 5*TILEWIDTH];

    float p_value = 0.0f;
    float p_value_2 = 0.0f;
    float p_value_3 = 0.0f;
    float p_value_4 = 0.0f;
    float p_value_5 = 0.0f;
    float p_value_6 = 0.0f;
    for (int m =0; m < N / TILEWIDTH; ++m){
        s_A[threadIdx.y][threadIdx.x] = preload_a;
        s_B[threadIdx.y][threadIdx.x] = preload_b;

        s_B[threadIdx.y][threadIdx.x + TILEWIDTH] = preload_b_2;
        s_B[threadIdx.y][threadIdx.x + 2*TILEWIDTH] = preload_b_3;
        s_B[threadIdx.y][threadIdx.x + 3*TILEWIDTH] = preload_b_4;
        s_B[threadIdx.y][threadIdx.x + 4*TILEWIDTH] = preload_b_5;
        s_B[threadIdx.y][threadIdx.x + 5*TILEWIDTH] = preload_b_6;
        // s_A[threadIdx.y][threadIdx.x] = A[row*N + (m * TILEWIDTH + threadIdx.x)];
        // s_B[threadIdx.y][threadIdx.x] = B[(m * TILEWIDTH + threadIdx.y)*N + col];
        __syncthreads();
        if (m+1 < N / TILEWIDTH - 1){
            preload_a = (row < N && ((m+1) * TILEWIDTH + threadIdx.x) < N) ? A[row * N + ((m+1) * TILEWIDTH + threadIdx.x)] : 0.0f;
            preload_b = (((m+1) * TILEWIDTH + threadIdx.y) < N && col < N) ? B[((m+1) * TILEWIDTH + threadIdx.y)*N + col] : 0.0f;

            preload_b_2 = (((m+1) * TILEWIDTH + threadIdx.y) < N && col + TILEWIDTH < N) ? B[((m+1) * TILEWIDTH + threadIdx.y)*N + col + TILEWIDTH] : 0.0f;
            preload_b_3 = (((m+1) * TILEWIDTH + threadIdx.y) < N && col + 2*TILEWIDTH < N) ? B[((m+1) * TILEWIDTH + threadIdx.y)*N + col + 2*TILEWIDTH] : 0.0f;
            preload_b_4 = (((m+1) * TILEWIDTH + threadIdx.y) < N && col + 3*TILEWIDTH < N) ? B[((m+1) * TILEWIDTH + threadIdx.y)*N + col + 3*TILEWIDTH] : 0.0f;
            preload_b_5 = (((m+1) * TILEWIDTH + threadIdx.y) < N && col + 4*TILEWIDTH < N) ? B[((m+1) * TILEWIDTH + threadIdx.y)*N + col + 4*TILEWIDTH] : 0.0f;
            preload_b_6 = (((m+1) * TILEWIDTH + threadIdx.y) < N && col + 5*TILEWIDTH < N) ? B[((m+1) * TILEWIDTH + threadIdx.y)*N + col + 5*TILEWIDTH] : 0.0f;
        }
        #pragma unroll
        for (int k = 0; k < TILEWIDTH; ++k){
            p_value += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
            p_value_2 += s_A[threadIdx.y][k] * s_B[k][threadIdx.x + TILEWIDTH];
            p_value_3 += s_A[threadIdx.y][k] * s_B[k][threadIdx.x + 2*TILEWIDTH];
            p_value_4 += s_A[threadIdx.y][k] * s_B[k][threadIdx.x + 3*TILEWIDTH];
            p_value_5 += s_A[threadIdx.y][k] * s_B[k][threadIdx.x + 4*TILEWIDTH];
            p_value_6 += s_A[threadIdx.y][k] * s_B[k][threadIdx.x + 5*TILEWIDTH];
        }
        __syncthreads();
    }
    if (row<N && col < N) {
        result[row * N + col] = p_value;
    }
    if (row<N && col + TILEWIDTH < N) {
        result[row * N + col + TILEWIDTH] = p_value_2;
    }
    if (row<N && col + 2*TILEWIDTH < N) {
        result[row * N + col + 2*TILEWIDTH] = p_value_3;
    }
    if (row<N && col + 3*TILEWIDTH < N) {
        result[row * N + col + 3*TILEWIDTH] = p_value_4;
    }
    if (row<N && col + 4*TILEWIDTH < N) {
        result[row * N + col + 4*TILEWIDTH] = p_value_5;
    }
    if (row<N && col + 5*TILEWIDTH < N) {
        result[row * N + col + 5*TILEWIDTH] = p_value_6;
    }
}
#define RADIX_BITS_4 4
#define RADIX_4      16
#define RADIX_BITS 3
#define RADIX      8

__device__ __forceinline__ unsigned ld_relaxed(const volatile unsigned* p) { return *p; }

// dynamic shared at launch:
//   shm = sizeof(unsigned) * ((blockDim.x + 1) * RADIX)
__global__ void CountRadixSort(int* array, unsigned* g_offset, int N, int bits) {

    const int T    = blockDim.x;
    const int L = gridDim.x;
    const int tid  = threadIdx.x;
    const int bid = blockIdx.x;
    const int row  = (bid * T + tid + 1) * (RADIX+1);
    const int chunk  = (N + T*L - 1) / (T*L);
    const int start  = tid * chunk + bid * chunk * T;
    const int end    = min(start + chunk, N);

    // ---- per-thread counters for 256 bins ----
    unsigned c[RADIX];
    #pragma unroll
    for (int b = 0; b < RADIX; ++b) c[b] = 0u;

    for (int i = start; i < end; ++i) {
        unsigned k = (((unsigned)array[i]) >> bits) & (RADIX - 1u);
        ++c[k];
    }

    // write my row into thread_offset (row-major)
    unsigned* rowPtr = g_offset + row;
    for (int b = 0; b < RADIX; ++b) rowPtr[b] = c[b];
    __threadfence();

    volatile unsigned* prevPtr = reinterpret_cast<volatile unsigned*>(rowPtr-1);
    while (ld_relaxed(prevPtr) < 2u) {
        // optional backoff: __nanosleep(32);
    }

    volatile unsigned* prevRow = rowPtr - (RADIX + 1);
    #pragma unroll
    for (int b = 0; b < RADIX; ++b) {
        rowPtr[b] += prevRow[b];
    }
    // Publish my row as G
    __threadfence();              // counts visible before status
    rowPtr[RADIX] = 2u;

    if (tid == T-1 && bid == L-1) {
        // Last thread: write total count to row 0
        unsigned sum = 0;
        for (int b = 0; b < RADIX; ++b) {
            unsigned v = rowPtr[b];
            rowPtr[b] = sum;
            sum += v;   
        }
    }
}


__global__ void ScatterRadixSort(int* array, unsigned* g_offset, int* final, int N , int bits){
    
    const int T    = blockDim.x;
    const int L = gridDim.x;
    const int tid  = threadIdx.x;
    const int bid = blockIdx.x;
    const int row  = (bid * T + tid) * (RADIX+1);
    const int chunk  = (N + T*L - 1) / (T*L);
    const int start  = tid * chunk + bid * chunk * T;
    const int end    = min(start + chunk, N);
    int last_row = (T*L) * (RADIX+1);

    // ---- per-thread counters for 256 bins ----
   
    for (int i = start; i < end; ++i) {
        unsigned k = (((unsigned)array[i]) >> bits) & (RADIX - 1u);
        final[ (int)g_offset[last_row + (int)k] + (int)g_offset[row + (int)k] ] = array[i];
        g_offset[row + (int)k]++;
    }
}


extern "C" void launchRadixSort(int *input_array, int *output_array, int N, int max_val) {
    int *d_input;
    int *d_output;

    unsigned *global_counter;

    cudaMalloc(&d_input, sizeof(int) * N);
    cudaMalloc(&d_output, sizeof(int) * N);

    cudaMemcpy(d_input, input_array, sizeof(int) * N, cudaMemcpyHostToDevice);

    int bits = static_cast<int>(std::log2(max_val));
    int max_passes = (bits + RADIX_BITS - 1) / RADIX_BITS;
    int threadsPerBlock = min(256, N);
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    cudaMalloc(&global_counter, sizeof(unsigned) * (RADIX+1) * (threadsPerBlock * numBlocks + 1));
    cudaMemset(global_counter, 0, sizeof(unsigned) * (RADIX+1) * (threadsPerBlock * numBlocks + 1));
    cudaMemset(global_counter+RADIX, 2, sizeof(unsigned));
    for (int pass = 0; pass < max_passes; pass++){
        CountRadixSort<<<numBlocks, threadsPerBlock>>>(d_input, global_counter, N, pass * RADIX_BITS);
        // ScanRadixSort<<<1, threadsPerBlock, sizeof(unsigned) * (threadsPerBlock + 1) * RADIX>>>(global_counter, numBlocks * threadsPerBlock);
        ScatterRadixSort<<<numBlocks, threadsPerBlock>>>(d_input, global_counter, d_output, N, pass * RADIX_BITS);
        cudaMemcpy(d_input, d_output, sizeof(int) * N, cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(output_array, d_output, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

}


// A helper function you can call from C++
extern "C" void launchMatrixMul(float *d_A, float *d_B, float *d_result, int N, int threadsPerBlockX, int threadsPerBlockY) {
    float *A;
    float *B;
    float *result;

    cudaMalloc(&A, N * N * sizeof(float));
    cudaMalloc(&B, N * N * sizeof(float));
    cudaMalloc(&result, N * N * sizeof(float));

    cudaMemcpy(A, d_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, d_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(threadsPerBlockX, threadsPerBlockY);
    dim3 numBlocks((N + threadsPerBlock.x*X_MULT - 1) / (threadsPerBlock.x * X_MULT),
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMul<<<numBlocks, threadsPerBlock>>>(A, B, result, N);
    cudaDeviceSynchronize();
    cudaMemcpy(d_result, result, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A);
    cudaFree(B);
    cudaFree(result);
}


// __global__ void ScanRadixSort(unsigned* g_offset,  int g_offset_N){
    
//     // scan launched with one grid block, 
//     // g_offset_N = numBlocks * (threadsPerBlock) , so true "N" is( g_offset_N+1) * RADIX
//     extern __shared__ unsigned thread_offset[];

//     const int T    = blockDim.x;
//     const int tid  = threadIdx.x;
//     const int row  = tid * RADIX;
//     const int chunk  = ((g_offset_N + T - 1) / T) * RADIX;
//     int start  = tid * chunk;
//     int end    = min(start + chunk, (g_offset_N)*RADIX);
//     //first do in-lane scan
//     unsigned c[RADIX];
//     #pragma unroll
//     for (int b = 0; b < RADIX; ++b) c[b] = 0u;

//     for (int i = start; i < end; i+= RADIX) {
//         for (int b = 0; b < RADIX; ++b) {
//             unsigned z = g_offset[i+b];
//             g_offset[i+b] = c[b];
//             c[b] += z;
//         }
//     }

//     for (int b = 0; b < RADIX; ++b) {
//         thread_offset[row +RADIX + b] = c[b];
//     }

//     __syncthreads();

//     if (tid == 0) {
//         int base = 0;
//         for (int b = 0; b < RADIX; ++b) {
//             thread_offset[base + b] = 0;
//         }
//     }
//     __syncthreads();

//     for (int k = 1; k < T; k <<= 1) {
//         if (tid < T - k) {
//             int base = tid * RADIX;
//             for (int b = 0; b < RADIX; ++b) c[b] = thread_offset[base + b];
//         }
//         __syncthreads();
//         if (tid < T - k) {
//             int base = (tid + k) * RADIX;
//             for (int b = 0; b < RADIX; ++b) thread_offset[base + b] += c[b];
//         }
//         __syncthreads();
//     }

//     if (tid == 0){
//         int base = T * RADIX;
//         int prev_row = (T-1) * RADIX;

//         int sum = 0;

//         for (int b = 0; b < RADIX; ++b) {
//             unsigned v = thread_offset[base + b] + thread_offset[prev_row + b];
//             thread_offset[base + b] = sum;
//             sum += v;
//         }
//     }
//     __syncthreads();

//     for (int i = start; i < end; i+= RADIX) {
//         for (int b = 0; b < RADIX; ++b) {
//             g_offset[i+b] += thread_offset[(tid)*RADIX + b];
//         }
//     }

//     if (tid == 0) {
//         int base = (g_offset_N) * RADIX;
//         for (int b = 0; b < RADIX; ++b) {
//             g_offset[base + b] = thread_offset[(T)*RADIX + b];
//         }
//     }
// }

// dynamic shared at launch: shm = sizeof(unsigned) * (blockDim.x * RADIX)
__global__ void RadixSort_4(int* array, int* final, int N, int bits) {
    // __shared__ unsigned pre_count[RADIX];
    extern __shared__ unsigned thread_offset[];
    // for (int k = threadIdx.x; k < RADIX; k += blockDim.x) pre_count[k] = 0u;
    // __syncthreads();

    const int T    = blockDim.x;
    const int tid  = threadIdx.x;
    const int row  = tid * RADIX_4;
    const int chunk  = (N + T - 1) / T;
    const int start  = tid * chunk;
    const int end    = min(start + chunk, N);

    // ---- scalarized per-thread counters (branchless) ----
    unsigned c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0,c7=0,
             c8=0,c9=0,c10=0,c11=0,c12=0,c13=0,c14=0,c15=0;

    for (int i = start; i < end; ++i) {
        unsigned k = (((unsigned)array[i]) >> bits) & 0xFu;
        c0  += (k == 0);   c1  += (k == 1);   c2  += (k == 2);   c3  += (k == 3);
        c4  += (k == 4);   c5  += (k == 5);   c6  += (k == 6);   c7  += (k == 7);
        c8  += (k == 8);   c9  += (k == 9);   c10 += (k == 10);  c11 += (k == 11);
        c12 += (k == 12);  c13 += (k == 13);  c14 += (k == 14);  c15 += (k == 15);
    }

    // write my row into thread_offset (row-major)
    unsigned* rowPtr = thread_offset + row;
    rowPtr[0]=c0;  rowPtr[1]=c1;  rowPtr[2]=c2;  rowPtr[3]=c3;
    rowPtr[4]=c4;  rowPtr[5]=c5;  rowPtr[6]=c6;  rowPtr[7]=c7;
    rowPtr[8]=c8;  rowPtr[9]=c9;  rowPtr[10]=c10; rowPtr[11]=c11;
    rowPtr[12]=c12;rowPtr[13]=c13;rowPtr[14]=c14; rowPtr[15]=c15;

    __syncthreads();
    
    unsigned v0,v1,v2,v3,v4,v5,v6,v7,
             v8,v9,v10,v11,v12,v13,v14,v15;
    
    if (tid < blockDim.x) {
        int base = tid * RADIX_4;
        v0 = thread_offset[base+0];
        v1 = thread_offset[base+1];
        v2 = thread_offset[base+2];
        v3 = thread_offset[base+3];
        v4 = thread_offset[base+4];
        v5 = thread_offset[base+5];
        v6 = thread_offset[base+6];
        v7 = thread_offset[base+7];
        v8 = thread_offset[base+8];
        v9 = thread_offset[base+9];
        v10= thread_offset[base+10];
        v11= thread_offset[base+11];
        v12= thread_offset[base+12];
        v13= thread_offset[base+13];
        v14= thread_offset[base+14];
        v15= thread_offset[base+15];
    }
    __syncthreads();
    if (tid < blockDim.x) {
        int base = (tid+1) * RADIX_4;
        thread_offset[base+0] = v0;
        thread_offset[base+1] = v1;
        thread_offset[base+2] = v2;
        thread_offset[base+3] = v3;
        thread_offset[base+4] = v4;
        thread_offset[base+5] = v5;
        thread_offset[base+6] = v6;
        thread_offset[base+7] = v7;
        thread_offset[base+8] = v8;
        thread_offset[base+9] = v9;
        thread_offset[base+10] = v10;
        thread_offset[base+11] = v11;
        thread_offset[base+12] = v12;
        thread_offset[base+13] = v13;
        thread_offset[base+14] = v14;
        thread_offset[base+15] = v15;
    }
    if (tid == 0) {
        int base = 0;
        thread_offset[base+0] = 0;
        thread_offset[base+1] = 0;
        thread_offset[base+2] = 0;
        thread_offset[base+3] = 0;
        thread_offset[base+4] = 0;
        thread_offset[base+5] = 0;
        thread_offset[base+6] = 0;
        thread_offset[base+7] = 0;
        thread_offset[base+8] = 0;
        thread_offset[base+9] = 0;
        thread_offset[base+10]= 0;
        thread_offset[base+11]= 0;
        thread_offset[base+12]= 0;
        thread_offset[base+13]= 0;
        thread_offset[base+14]= 0;
        thread_offset[base+15]= 0;
    }
    __syncthreads();
    for (int k = 1; k < T; k*=2) {
        if (tid < blockDim.x-k) {
            int base = tid * RADIX_4;
            v0 = thread_offset[base+0];
            v1 = thread_offset[base+1];
            v2 = thread_offset[base+2];
            v3 = thread_offset[base+3];
            v4 = thread_offset[base+4];
            v5 = thread_offset[base+5];
            v6 = thread_offset[base+6];
            v7 = thread_offset[base+7];
            v8 = thread_offset[base+8];
            v9 = thread_offset[base+9];
            v10= thread_offset[base+10];
            v11= thread_offset[base+11];
            v12= thread_offset[base+12];
            v13= thread_offset[base+13];
            v14= thread_offset[base+14];
            v15= thread_offset[base+15];
        }
        __syncthreads();
        if (tid < blockDim.x-k) {
            int base = (tid+k) * RADIX_4;
            thread_offset[base+0] += v0;
            thread_offset[base+1] += v1;
            thread_offset[base+2] += v2;
            thread_offset[base+3] += v3;
            thread_offset[base+4] += v4;
            thread_offset[base+5] += v5;
            thread_offset[base+6] += v6;
            thread_offset[base+7] += v7;
            thread_offset[base+8] += v8;
            thread_offset[base+9] += v9;
            thread_offset[base+10]+= v10;
            thread_offset[base+11]+= v11;
            thread_offset[base+12]+= v12;
            thread_offset[base+13]+= v13;
            thread_offset[base+14]+= v14;
            thread_offset[base+15]+= v15;
        }
        __syncthreads();
    }
    if (tid == 0) {
        int base = T * RADIX_4;
        int prev_row = (T-1) * RADIX_4;
        v0 = thread_offset[base+0] + thread_offset[prev_row+0];
        v1 = thread_offset[base+1] + thread_offset[prev_row+1];
        v2 = thread_offset[base+2] + thread_offset[prev_row+2];
        v3 = thread_offset[base+3] + thread_offset[prev_row+3];
        v4 = thread_offset[base+4] + thread_offset[prev_row+4];
        v5 = thread_offset[base+5] + thread_offset[prev_row+5];
        v6 = thread_offset[base+6] + thread_offset[prev_row+6];
        v7 = thread_offset[base+7] + thread_offset[prev_row+7];
        v8 = thread_offset[base+8] + thread_offset[prev_row+8];
        v9 = thread_offset[base+9] + thread_offset[prev_row+9];
        v10= thread_offset[base+10] + thread_offset[prev_row+10];
        v11= thread_offset[base+11] + thread_offset[prev_row+11];
        v12= thread_offset[base+12] + thread_offset[prev_row+12];
        v13= thread_offset[base+13] + thread_offset[prev_row+13];
        v14= thread_offset[base+14] + thread_offset[prev_row+14];
        v15= thread_offset[base+15] + thread_offset[prev_row+15];

        thread_offset[base+0] = 0;
        thread_offset[base+1] = v0;
        thread_offset[base+2] = v0 + v1;
        thread_offset[base+3] = v0 + v1 + v2;
        thread_offset[base+4] = v0 + v1 + v2 + v3;
        thread_offset[base+5] = v0 + v1 + v2 + v3 + v4;
        thread_offset[base+6] = v0 + v1 + v2 + v3 + v4 + v5;
        thread_offset[base+7] = v0 + v1 + v2 + v3 + v4 + v5 + v6;
        thread_offset[base+8] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
        thread_offset[base+9] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8;
        thread_offset[base+10] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9;
        thread_offset[base+11] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10;
        thread_offset[base+12] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10 + v11;
        thread_offset[base+13] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10 + v11 + v12;
        thread_offset[base+14] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10 + v11 + v12 + v13;
        thread_offset[base+15] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10 + v11 + v12 + v13 + v14;
    }
    __syncthreads();
    // scatter (unchanged pattern)
    int last_row = (T)*RADIX_4;
    int last_row_bef = (T-1)*RADIX_4;
    for (int i = start; i < end; ++i) {
        unsigned bin = (((unsigned)array[i]) >> bits) & 0xFu;
        final[ thread_offset[last_row + bin] + thread_offset[row + bin] ] = array[i];
        thread_offset[row + bin]++;  // per-thread "seen" in shared
    }
}

extern "C" void launchRadixSort_4(int *input_array, int *output_array, int N, int max_val, float* kernel_ms) {
    int *d_input;
    int *d_output;

    cudaMalloc(&d_input, sizeof(int) * N);
    cudaMalloc(&d_output, sizeof(int) * N);

    cudaMemcpy(d_input, input_array, sizeof(int) * N, cudaMemcpyHostToDevice);

    int bits = static_cast<int>(std::log2(max_val));
    int max_passes = (bits + RADIX_BITS_4 - 1) / RADIX_BITS_4;
    int threadsPerBlock = min(256, N);
    int numBlocks = 1;
    int memSize = sizeof(unsigned int) * RADIX_4 * (threadsPerBlock+1);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start);
    for (int pass = 0; pass < max_passes; pass++){
        RadixSort_4<<<numBlocks, threadsPerBlock, memSize>>>(d_input, d_output, N, pass * RADIX_BITS_4);
        cudaMemcpy(d_input, d_output, sizeof(int) * N, cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, ev_start, ev_stop);
    *kernel_ms = milliseconds;

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    
    cudaDeviceSynchronize();
    cudaMemcpy(output_array, d_output, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

}