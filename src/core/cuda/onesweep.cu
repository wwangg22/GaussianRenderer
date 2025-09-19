#include <cuda_runtime.h>
#include <iostream>
#include <cmath> 

#define RADIX_BITS 8
#define RADIX 256
__global__ void globalBinCounter(int* input_array, int* global_counter, int numPasses, int N){
    extern __shared__ int local_counter[]; //warps * RADIX * numPasses
    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x & 31;
    int numWarps = 8;
    int *start = local_counter + (warpId * (RADIX) * numPasses);

    for (int j = laneId; j < RADIX*numPasses; j += 32) {
        start[j] = 0;
    }
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        int v = input_array[i];
        #pragma unroll
        for (int p = 0; p < 4; ++p) {

            unsigned d = (v >> (p * RADIX_BITS)) & (RADIX - 1);
            atomicAdd(&start[p * RADIX + d], 1u);
            
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < numWarps*numPasses; i += blockDim.x) {
        int sum = 0;
        for (int j = 0; j < RADIX; j++) {
            int idx = i * RADIX + j;
            int c = local_counter[idx];
            local_counter[idx] = sum;
            sum += c;
        }
    }
    __syncthreads();



    for (int j = threadIdx.x; j < RADIX * numPasses; j += blockDim.x) {
        unsigned int sum = 0u;
        #pragma unroll
        for (int w = 0; w < numWarps; ++w) sum += local_counter[w * RADIX * numPasses + j];
        atomicAdd(&global_counter[j], sum);
    }

}
__device__ int gTileCounter;
#define TILE_SIZE 1024
__global__ void oneSweep(int* input_array, int* output_array, int* lookback, int* global_counter, int N, int shift) {
    __shared__ int shared_current_tile;
    if (threadIdx.x == 0) shared_current_tile = atomicAdd(&gTileCounter, 1);
    __syncthreads();
    int current_tile = shared_current_tile;
    const int numTiles   = (N + TILE_SIZE - 1) / TILE_SIZE;
    if (current_tile >= numTiles) return;
    __shared__ int local_offset[TILE_SIZE];
    __shared__ int counter_full[RADIX*(9)]; //( num warps + 1)* RADIX
    __shared__ int tile_offset[RADIX];

    

    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x & 31;
    int numWarps = 8;

    int chunk = TILE_SIZE / 256;
    int* counter = counter_full;

    int start = current_tile * TILE_SIZE + warpId * 32 * chunk + laneId;
    int end = min(N, current_tile * TILE_SIZE + (warpId+1) * 32 * chunk);

    for (int j = threadIdx.x; j < RADIX * (numWarps + 1); j += blockDim.x) counter_full[j] = 0;
    __syncthreads();

    for (int i = laneId + warpId * 32; i < TILE_SIZE; i += 256) {
        local_offset[i] = 0;
    }
    __syncthreads();
    unsigned mask[8];
    int pop;
    int offset = 0;
    for (int i = start; i < end; i += 32) {
        unsigned active = __activemask();
        int v = (input_array[i] >> (shift*RADIX_BITS)) & (RADIX - 1);
        //grab ballots from all threads in the warp
        for (int j = 0; j < 8; ++j) {
            mask[j] = __ballot_sync(active, (v >> j) & 1);
        }
        //find the threads that have same digit as me
        unsigned my_group = active;
        for (int j = 0; j < 8; ++j) {
            //i tried not to introduce any if else to prevent warp divergence
            unsigned bit   = (v >> j) & 1u;
            unsigned keep  = bit ? mask[j] : (active ^ mask[j]);  // complement within 'active'
            my_group      &= keep;
        }
        pop = __popc(my_group);
        int leader = __ffs(my_group) - 1;
        if (laneId == leader) {
            offset = atomicAdd(&counter[warpId * RADIX + v], pop);
        }
        offset = __shfl_sync(my_group, offset, leader);

        local_offset[i - current_tile * TILE_SIZE] = offset + __popc(my_group & ((1u << laneId) - 1));
    }
    __syncthreads();
    if (threadIdx.x < RADIX) {
        int sum = 0;
        #pragma unroll
        for (int w = 0; w < numWarps; ++w) {
            int c = counter[w * RADIX + threadIdx.x];   // this warp’s count
            counter[w * RADIX + threadIdx.x] = sum;           // exclusive prefix for this warp
            sum += c;
        }
        // store this tile’s *local* count per digit (for chaining)
        counter_full[numWarps * RADIX + threadIdx.x] = sum;   // tile_count[d]
    }
    __syncthreads();    
    
    // // if (j >= 0) counter_full[8*RADIX + j] += counter_full[7*RADIX + j];
    // // __syncthreads();
    unsigned flag;
    if (current_tile == 0) {
        flag = (1<<31); //first tile
    } else {
        flag = (1<<30);
    }

    for (int k = threadIdx.x; k < RADIX; k += blockDim.x) {
        atomicExch(&lookback[current_tile * (RADIX) + k], counter_full[8*RADIX + k] | flag);
        // lookback[current_tile * (RADIX) + k] = counter_full[8*RADIX + k] | flag;
    }
    __syncthreads();

    for (int k = threadIdx.x; k < RADIX; k += blockDim.x) {
            tile_offset[k] = 0;
    }
    __syncthreads();

    if (current_tile > 0){

        for (int k = threadIdx.x; k < RADIX; k += blockDim.x) {
            int prev_tile = (current_tile - 1) * RADIX;
            while (prev_tile >= 0){
                int flag = atomicAdd(&lookback[prev_tile + k], 0);
                bool global = ((flag >> 31) & 1) != 0;
                bool local  = ((flag >> 30) & 1) != 0;

                if (global){
                    tile_offset[k] += (flag & 0x3FFFFFFF);
                    break;
                } else if (local){
                    tile_offset[k] += (flag & 0x3FFFFFFF);
                    prev_tile -= RADIX;
                }else {
                    __nanosleep(64);
                }
            }

            // lookback[current_tile*(RADIX) + k] = (counter_full[8*RADIX + k] + tile_offset[k]) | (1 << 31);
            atomicExch(&lookback[current_tile*(RADIX) + k], (counter_full[8*RADIX + k] + tile_offset[k]) | (1 << 31));
        }
        __syncthreads();
    }

    
    __syncthreads();
    //start to scatter globally

    for (int i = start; i < end; i += 32) {
        int v = (input_array[i] >> (shift*RADIX_BITS)) & (RADIX - 1);
        int global_offset = global_counter[shift * RADIX + v];
        int pos = global_offset + tile_offset[v] + counter_full[warpId * RADIX + v] + local_offset[i - current_tile * TILE_SIZE];
        output_array[pos] = input_array[i];
    }

}

extern "C" void oneSweepSort(int* input, int* output, int N, int maxVal, float* kernel_ms){
    int* d_global_counter;
    int* d_lookback;
    int* d_input;
    int* d_output;
    cudaMalloc(&d_output, N * sizeof(int));
    int numPasses = 4;

    int BLOCK_SIZE = 256;
    int NUM_BLOCKS = 174;

    int zero = 0;

    int TOTAL_TILES = (N + TILE_SIZE - 1) / TILE_SIZE;
    int SHARED_MEMORY_SIZE = RADIX * numPasses * (BLOCK_SIZE/32) * sizeof(int);

    cudaMalloc(&d_global_counter, numPasses * (RADIX ) * sizeof(int));
    cudaMalloc(&d_lookback, TOTAL_TILES * (RADIX) * sizeof(int));
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMemset(d_global_counter, 0, numPasses * (RADIX) * sizeof(int));

    memcpy(output, input, N * sizeof(int));
    cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, input, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start);

    globalBinCounter<<<NUM_BLOCKS, BLOCK_SIZE, SHARED_MEMORY_SIZE>>>(d_input, d_global_counter, numPasses, N);
    for (int shift = 0; shift < numPasses; ++shift) {
        cudaMemcpyToSymbol(gTileCounter, &zero, sizeof(int));
        cudaMemset(d_lookback, 0, TOTAL_TILES * (RADIX) * sizeof(int));
        oneSweep<<<TOTAL_TILES, BLOCK_SIZE>>>(d_input, d_output, d_lookback, d_global_counter, N, shift);
        cudaMemcpy(d_input, d_output, N * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, ev_start, ev_stop);
    *kernel_ms = milliseconds;

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaMemcpy(output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_global_counter);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_lookback);
}