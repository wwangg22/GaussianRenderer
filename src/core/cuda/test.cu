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
#define RADIX_BITS 4
// assume threadDim > 1<<radix_bits
__global__ void RadixSort(int* array, int* final, int N, int bits) {
    // bits is sort of the index of the bits we are sorting on
    __shared__ unsigned int pre_count[(1 << RADIX_BITS)];
    extern __shared__ unsigned thread_offset[];

    for (int idx = threadIdx.x; idx < (1 << RADIX_BITS); idx += blockDim.x) pre_count[idx]=0;

    __syncthreads();


    int local_count[1 << RADIX_BITS] = {0};

    int chunk = N / blockDim.x + (N % blockDim.x != 0);

    int row = threadIdx.x * (1 << RADIX_BITS);

    const int start = threadIdx.x * chunk;
    const int end   = min(start + chunk, N);
    for (int i = start; i < end; ++i) {
        unsigned k = (unsigned)array[i] >> bits & 0xF;
        local_count[k]++;
    }
    for (int i = 0; i < (1 << RADIX_BITS); ++i){
        atomicAdd(&pre_count[i], local_count[i]);
        thread_offset[i + row] = local_count[i];
    }

    __syncthreads();
    if (threadIdx.x == 0){
        int sum = 0;
        int cur;
        for (int i = 0; i < (1 << RADIX_BITS); ++i){
            cur = pre_count[i];
            pre_count[i] = sum;
            sum += cur;
        }

        for (int i = 0; i < (1<<RADIX_BITS); ++i){
            local_count[i] = 0;
        }

        for (int i = 0; i < blockDim.x; ++i){
            for (int j = 0; j < (1 << RADIX_BITS); ++j){
                int carry = thread_offset[j + (1 << RADIX_BITS) * i];
                thread_offset[j + (1 << RADIX_BITS) * i] = local_count[j];
                local_count[j] += carry;

            }
        }
    }
    __syncthreads();

    for (int i = start; i < end; ++i) {
        unsigned int bin = (unsigned)array[i] >> bits & 0xF;
        final[pre_count[bin] + thread_offset[bin + row]] = array[i];
        thread_offset[bin + row]++;
    }
}

extern "C" void launchRadixSort(int *input_array, int *output_array, int N, int max_val) {
    int *d_input;
    int *d_output;

    cudaMalloc(&d_input, sizeof(int) * N);
    cudaMalloc(&d_output, sizeof(int) * N);

    cudaMemcpy(d_input, input_array, sizeof(int) * N, cudaMemcpyHostToDevice);

    int bits = static_cast<int>(std::log2(max_val));
    int max_passes = (bits + RADIX_BITS - 1) / RADIX_BITS;
    int threadsPerBlock = min(256, N);
    int numBlocks = 1;
    for (int pass = 0; pass < max_passes; pass++){
        RadixSort<<<numBlocks, threadsPerBlock, sizeof(unsigned int) * (1 << RADIX_BITS) * threadsPerBlock>>>(d_input, d_output, N, pass * RADIX_BITS);
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
