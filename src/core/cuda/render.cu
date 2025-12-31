#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/cub.cuh>
#include <cstdint>
#include "gaussians.hpp"
#include "camera.hpp"
#include "render.cuh"
#include "math.cuh"

static __global__ void globalBinCounter(lightWeightGaussian* d_in, int* d_global_counter, int numPasses, int N) {
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
        uint64_t v = d_in[i].radix_id;
        for (int p = 0; p < numPasses; ++p) {
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
        atomicAdd(&d_global_counter[j], sum);
    }
}

static __global__ void oneSweep(lightWeightGaussian* d_in, lightWeightGaussian* d_out, int* lookback, int* global_counter, int N, int shift) {
    // __shared__ int shared_current_tile;
    // if (threadIdx.x == 0) shared_current_tile = atomicAdd(&gTileCounter, 1);
    // __syncthreads();
    // int current_tile = shared_current_tile;
    int current_tile = blockIdx.x;
    const int numTiles   = (N + TILE_SIZE - 1) / TILE_SIZE;
    if (current_tile >= numTiles) return;
    // __shared__ int local_offset[TILE_SIZE];
    // __shared__ int local_value[TILE_SIZE];
    __shared__ int counter_full[RADIX*(9)]; //( num warps + 1)* RADIX
    __shared__ int tile_offset[RADIX];


    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x & 31;
    int numWarps = blockDim.x / 32;

    int chunk = TILE_SIZE / 256;
    int local_offset[8]; //TILE_SIZE  / 256
    lightWeightGaussian local_value[8]; //  TILE_SIZE  / 256
    int* counter = counter_full;

    int start = current_tile * TILE_SIZE + warpId * 32 * chunk + laneId;
    int end =  current_tile * TILE_SIZE + (warpId+1) * 32 * chunk;

    for (int j = threadIdx.x; j < RADIX * (numWarps + 1); j += blockDim.x) counter_full[j] = 0;
    __syncthreads();

    // for (int i = laneId + warpId * 32; i < TILE_SIZE; i += blockDim.x) {
    //     local_offset[i] = 0;
    // }
    #pragma unroll
    for (int i =0; i < 4; ++i){
        local_offset[i] = 0;
    }
    __syncthreads();
    unsigned mask[RADIX_BITS];
    int pop;
    int offset = 0;
    int part;
    for (int i = start; i < end; i += 32) {
        if (i >= N) part  = 0;
        else part = 1;
        unsigned active = __ballot_sync(0xFFFFFFFF, part);
        if (!part) continue;
        lightWeightGaussian lwg = d_in[i];
        uint64_t val = lwg.radix_id;
        // local_value[i - current_tile * TILE_SIZE] = val;
        local_value[(i - start) / 32] = lwg;
        int v = (val >> (shift*RADIX_BITS)) & (RADIX - 1);

        unsigned my_group = __match_any_sync(active,v);
        pop = __popc(my_group);
        int leader = __ffs(my_group) - 1;
        if (laneId == leader) {
            offset = counter[warpId * RADIX + v];
            counter[warpId * RADIX + v] += pop;
        }
        offset = __shfl_sync(my_group, offset, leader);

        // local_offset[i - current_tile * TILE_SIZE] = offset + __popc(my_group & ((1u << laneId) - 1));
        local_offset[(i - start) / 32] = offset + __popc(my_group & ((1u << laneId) - 1));
    }
    __syncthreads();
    if (threadIdx.x < RADIX) {
        int sum = 0;
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
        __threadfence();
        // atomicExch(&lookback[current_tile * (RADIX) + k], counter_full[8*RADIX + k] | flag);
        lookback[current_tile * (RADIX) + k] = counter_full[8*RADIX + k] | flag;
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
                // int flag = atomicAdd(&lookback[prev_tile + k], 0);
                __threadfence();
                int flag = lookback[prev_tile + k];
                bool global = ((flag >> 31) & 1) != 0;
                bool local  = ((flag >> 30) & 1) != 0;

                if (global){
                    tile_offset[k] += (flag & 0x3FFFFFFF);
                    break;
                } else if (local){
                    tile_offset[k] += (flag & 0x3FFFFFFF);
                    prev_tile -= RADIX;
                }else {
                }
            }
            __threadfence();
            lookback[current_tile*(RADIX) + k] = (counter_full[8*RADIX + k] + tile_offset[k]) | (1 << 31);
            // atomicExch(&lookback[current_tile*(RADIX) + k], (counter_full[8*RADIX + k] + tile_offset[k]) | (1 << 31));
        }
        __syncthreads();
    }
    
    for (int k = threadIdx.x; k < RADIX; k += blockDim.x) {
        tile_offset[k] += global_counter[shift * RADIX + k];
    }
        //start to scatter globally
    __syncthreads();

    for (int i = start; i < end; i += 32) {
        if (i >= N) continue;
        // int v = (local_value[i - current_tile * TILE_SIZE] >> (shift*RADIX_BITS)) & (RADIX - 1);
        int v = (local_value[(i - start) / 32].radix_id >> (shift*RADIX_BITS)) & (RADIX - 1);
        // int pos = tile_offset[v] + counter_full[warpId * RADIX + v] + local_offset[i - current_tile * TILE_SIZE];
        // output_array[pos] = local_value[i - current_tile * TILE_SIZE];
        int pos = tile_offset[v] + counter_full[warpId * RADIX + v] + local_offset[(i - start) / 32];
        d_out[pos] = local_value[(i - start) / 32];
    }

}

extern "C" void oneSweep3DGaussianSort(lightWeightGaussian* d_in, 
                                       int N, 
                                       int num_bits,
                                       float* kernel_ms) {
    
    int* d_global_counter;
    int* d_lookback;
    lightWeightGaussian* d_input;
    lightWeightGaussian* d_output;
    cudaMalloc(&d_output, N * sizeof(lightWeightGaussian));
    int numPasses = (num_bits + 7) / 8;

    int BLOCK_SIZE = 256;
    int NUM_BLOCKS = 174;

    int zero = 0;

    int TOTAL_TILES = (N + TILE_SIZE - 1) / TILE_SIZE;
    int SHARED_MEMORY_SIZE = RADIX * numPasses * (BLOCK_SIZE/32) * sizeof(int);

    cudaMalloc(&d_global_counter, numPasses * (RADIX ) * sizeof(int));
    cudaMalloc(&d_lookback, TOTAL_TILES * (RADIX) * sizeof(int));
    cudaMalloc(&d_input, N * sizeof(lightWeightGaussian));
    cudaMemset(d_global_counter, 0, numPasses * (RADIX) * sizeof(int));

    cudaMemcpy(d_input, d_in, N * sizeof(lightWeightGaussian), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, d_in, N * sizeof(lightWeightGaussian), cudaMemcpyHostToDevice);
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start);
    lightWeightGaussian* in = d_input;
    lightWeightGaussian* out = d_output;

    globalBinCounter<<<NUM_BLOCKS, BLOCK_SIZE, SHARED_MEMORY_SIZE>>>(d_input, d_global_counter, numPasses, N);
     cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("globalBinCounter launch error: %s\n", cudaGetErrorString(err));
    }

    // Then check execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("globalBinCounter runtime error: %s\n", cudaGetErrorString(err));
    }
    for (int shift = 0; shift < numPasses; ++shift) {
        // cudaMemcpyToSymbol(gTileCounter, &zero, sizeof(int));
        cudaMemset(d_lookback, 0, TOTAL_TILES * (RADIX) * sizeof(int));
        oneSweep<<<TOTAL_TILES, BLOCK_SIZE>>>(in, out, d_lookback, d_global_counter, N, shift);
        std::swap(in, out);
        // cudaMemcpy(d_input, d_output, N * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, ev_start, ev_stop);
    *kernel_ms = milliseconds;

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    lightWeightGaussian* result = (numPasses & 1) ? in : out; 
    cudaMemcpy(d_in, result, N * sizeof(lightWeightGaussian), cudaMemcpyDeviceToHost);

    cudaFree(d_global_counter);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_lookback);
}

static __global__ void renderGaussians(float* out_pixels, int* tile_id_offset, Gaussian* gaussians, lightWeightGaussian* sorted_gaussians,
                int height_stride, int width_stride, int tile_W, int tile_H,int num_tile_x, int num_tile_y) {
    extern __shared__ float buf[];
    float* shared_T = buf; // tile_info->height_stride * tile_info->width_stride
    float* shared_rgb = buf + height_stride * width_stride; // 3 * tile_info->height_stride * tile_info->width_stride
    const int W = tile_W, H = tile_H;
    const int xs = width_stride, ys = height_stride;
    const int ny = num_tile_y, nx = num_tile_x;
    const int block_size = xs*ys;
    __shared__ Gaussian cur_gauss;
    __shared__ lightWeightGaussian cur_lwg;
    // initialize shared memory
    for (int j = threadIdx.x; j < block_size; j += blockDim.x) {
        shared_T[j] = 1.0f;
        for (int c = 0; c < 3; ++c) {
            shared_rgb[c*block_size + j] = 0.0f;
        }
    }
    __syncthreads();
    for (int grid_id = blockIdx.x; grid_id < ny * nx; grid_id += gridDim.x) {

        int tile_x = grid_id % nx;
        int tile_y = grid_id / nx;

        int x_offset = tile_x * xs;
        int y_offset = tile_y * ys;

        int tile_offset_end = tile_id_offset[grid_id];
        int tile_offset_start;

        if (grid_id == 0) {
            tile_offset_start = 0;
        } else {
            tile_offset_start = tile_id_offset[grid_id - 1];
        }

        for (int idx = tile_offset_start; idx < tile_offset_end; ++idx) {
            if (threadIdx.x == 0) {
                cur_lwg = sorted_gaussians[idx];
                cur_gauss = gaussians[cur_lwg.gaussian_id];
            }
            __syncthreads();

            int px_x = cur_gauss.px_x;
            int px_y = cur_gauss.px_y;

            int aabb_xmin = cur_gauss.aabb[0];
            int aabb_ymin = cur_gauss.aabb[1];
            int aabb_xmax = cur_gauss.aabb[2];
            int aabb_ymax = cur_gauss.aabb[3];

            float inv_covar[4];
            inv_covar[0] = cur_gauss.inv_covar[0];
            inv_covar[1] = cur_gauss.inv_covar[1];
            inv_covar[2] = cur_gauss.inv_covar[2];
            inv_covar[3] = cur_gauss.inv_covar[3];

            for (int j = threadIdx.x; j < block_size; j += blockDim.x) {
                int global_x = j % xs + x_offset;
                int global_y = j / xs + y_offset;
                if (global_x >= W || global_y >= H) continue;
                if (global_x < aabb_xmin || global_x > aabb_xmax || global_y < aabb_ymin || global_y > aabb_ymax) continue;
                if (shared_T[j] < 1e-3f) continue;
                float dx = (static_cast<float>(global_x) - static_cast<float>(px_x));
                float dy = (static_cast<float>(global_y) - static_cast<float>(px_y));

                float md2 = dx * (inv_covar[0]*dx + inv_covar[1]*dy) + dy * (inv_covar[2]*dx + inv_covar[3]*dy);
                float opacity = cur_gauss.opacity * expf(-0.5f * md2);
                opacity = fminf(opacity, 0.99f);
                if (opacity < 1e-3f) continue;

                for (int c = 0; c < 3; ++c) {
                    shared_rgb[c*block_size + j] += cur_gauss.color[c] * opacity * shared_T[j];
                }
                shared_T[j] *= (1.0f - opacity);
            }
           __syncthreads();

        }

        __syncthreads();

        for (int j = threadIdx.x; j < block_size; j += blockDim.x) {
            int gx = x_offset + (j % xs);
            int gy = y_offset + (j / xs);

            if (gx < W && gy < H) {
                int out_idx = gy * W + gx;
                for (int c = 0; c < 3; c++) {
                    out_pixels[c*(H * W) + out_idx] += shared_rgb[c*block_size + j];
                    shared_rgb[c*block_size + j] = 0.0f;
                }
            } else {
                // still reset shared memory even if pixel is out of bounds
                for (int c = 0; c < 3; c++) shared_rgb[c*block_size + j] = 0.0f;
            }
            shared_T[j] = 1.0f;
        }
        __syncthreads();
        
    }
}

const __device__ float SH_C0 = 0.28209479177387814f;
const __device__ float SH_C1 = 0.4886025119029199f;
const __device__ float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
const __device__ float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};
static __global__ void cullGaussians(Gaussian* d_gaussians,
    Gaussian* d_output_gaussians,
    int num_gaussians,
    Camera cam,
    int* d_culled_count,
    int* threadblock_counts,
    const float treshold) {
    // planes contains the normal vectors for the 6 planes (24,)
    // stored as [x,y,z, offset]
    float* planes = cam.plane_normals;

    // we will first use d_output_gaussians as a counter
    // for the different thread blocks

    // idea: each threadblock will increment its own lightweightgaussian
    // and then we will do a prefix sum to get the final indices
    __shared__ int local_count;
    __shared__ int offset;
    extern __shared__ int shared_data[];
    if (threadIdx.x == 0){
        local_count = 0;
        offset = 0;
    }
    __syncthreads();

    int width = blockDim.x * gridDim.x;
    int stride = (num_gaussians + width - 1) / width;

    int start = stride * blockIdx.x * blockDim.x + threadIdx.x;
    int end = min(stride * (blockIdx.x + 1) * blockDim.x, num_gaussians);

    for (int idx = start; idx < end; idx += blockDim.x) {
        Gaussian g = d_gaussians[idx];
        int i;
        for (i = 0; i < 6; ++i) {
            float* normal = planes + i * 4;
            float xyz[3] = {g.x, g.y, g.z};
            float dot;
            dotProduct_cuda(xyz, normal, dot);
            if (dot + planes[i*4+3] < -treshold){
                break;
            }
        }
        if (i == 6) {
            int l_offset = atomicAdd(&local_count, 1);
        // index = threadIdx.x 
            shared_data[l_offset] = idx;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0){
        int flag;
        if (blockIdx.x == 0){
            flag = (1<<31); //first block
        } else {
            flag = (1<<30);
        }
        threadblock_counts[blockIdx.x] = offset+local_count | flag;
        int prev = blockIdx.x - 1;
        while (prev >= 0) {
            __threadfence();
            int flag = threadblock_counts[prev];
            bool global = ((flag >> 31) & 1) != 0;
            bool local = ((flag >> 30) & 1) != 0;
            if (global){
                offset += (flag & 0x3FFFFFFF);
                break;
            } else if (local){
                offset += (flag & 0x3FFFFFFF);
                prev -= 1;
            }
        }
        __threadfence();
        threadblock_counts[blockIdx.x] = offset+local_count | (1 << 31);

        if (blockIdx.x == gridDim.x - 1){
            *d_culled_count = offset + local_count;
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < local_count; i += blockDim.x) {
        d_output_gaussians[offset + i] = d_gaussians[shared_data[i]];
    }
}
static __global__ void advancedCullGaussians(Gaussian* d_gaussians,
    Gaussian* d_output_gaussians,
    Gaussian* d_tmp,
    int num_gaussians,
    Camera cam,
    int* d_culled_count,
    int* threadblock_counts){

    __shared__ int local_count;
    __shared__ int offset;
    extern __shared__ int shared_data[];
    int deg = 3;
    float* V = cam.V_matrix;
    float* P = cam.P_matrix;

    int width = blockDim.x * gridDim.x;
    int stride = (num_gaussians + width - 1) / width;

    int start = stride * blockIdx.x * blockDim.x + threadIdx.x;
    int end = min(stride * (blockIdx.x + 1) * blockDim.x, num_gaussians);
    if (threadIdx.x == 0){
        local_count = 0;
        offset = 0;
    }
    __syncthreads();

    for (int idx = start; idx < end; idx += blockDim.x) {
        Gaussian gauss = d_gaussians[idx];
        float viewing_dir[3] = {gauss.x - cam.position[0],
                                gauss.y - cam.position[1],
                                gauss.z - cam.position[2]};
        normalize_cuda(viewing_dir);
        for (int i = 0; i <3; i++) gauss.color[i] = gauss.sh[i] * SH_C0; 

        if (deg > 0) {
            float x = viewing_dir[0];
            float y = viewing_dir[1];
            float z = viewing_dir[2];

            for (int i = 0; i < 3; i++){
                gauss.color[i] += SH_C1 * z * gauss.sh[2*3 + i];
                gauss.color[i] -= SH_C1 * y * gauss.sh[3 + i];
                gauss.color[i] -= SH_C1 * x * gauss.sh[3*3 + i];
            }
            if (deg > 1)
            {
                float xx = x * x, yy = y * y, zz = z * z;
                float xy = x * y, yz = y * z, xz = x * z;
                for (int i = 0; i < 3; i++) {
                    gauss.color[i] += SH_C2[0] * xy * gauss.sh[4*3 + i];
                    gauss.color[i] += SH_C2[1] * yz * gauss.sh[5*3 + i];
                    gauss.color[i] += SH_C2[2] * (2.0f * zz - xx - yy) * gauss.sh[6*3 + i];
                    gauss.color[i] += SH_C2[3] * xz * gauss.sh[7*3 + i];
                    gauss.color[i] += SH_C2[4] * (xx - yy) * gauss.sh[8*3 + i];
                }
            }
        }
        for (int i = 0; i <3; i++){
            gauss.color[i] += 0.5f;
            gauss.color[i] = fminf(fmaxf(gauss.color[i], 0.0f), 1.0f);
        }
        float old_xyz[4] = {gauss.x, gauss.y, gauss.z, 1.0f};
        float new_xyz[4];
        float tmp_xyz[4];

        matVecMul4D_cuda(V, old_xyz, tmp_xyz);
        gauss.X = tmp_xyz[0];
        gauss.Y = tmp_xyz[1];
        gauss.Z = tmp_xyz[2];

        matVecMul4D_cuda(P, tmp_xyz, new_xyz);
        new_xyz[0] = new_xyz[0] / new_xyz[3];
        new_xyz[1] =  new_xyz[1] /  new_xyz[3];
        new_xyz[2] =  new_xyz[2] /  new_xyz[3];

        gauss.x = new_xyz[0];
        gauss.y = new_xyz[1];
        gauss.z = new_xyz[2];
        if (tmp_xyz[2] >= 0 || new_xyz[2] < -1.0f || new_xyz[2] > 1.0f) {
            continue;
        }
        else {
            d_tmp[idx] = gauss;
            int count = atomicAdd(&local_count, 1);
            shared_data[count] = idx;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0){
        int flag;
        if (blockIdx.x == 0){
            flag = (1<<31); //first block
        } else {
            flag = (1<<30);
        }
        threadblock_counts[blockIdx.x] = offset+local_count | flag;
        int prev = blockIdx.x - 1;
        while (prev >= 0) {
            __threadfence();
            int flag = threadblock_counts[prev];
            bool global = ((flag >> 31) & 1) != 0;
            bool local = ((flag >> 30) & 1) != 0;
            if (global){
                offset += (flag & 0x3FFFFFFF);
                break;
            } else if (local){
                offset += (flag & 0x3FFFFFFF);
                prev -= 1;
            }
        }
        __threadfence();
        threadblock_counts[blockIdx.x] = offset+local_count | (1 << 31);

        if (blockIdx.x == gridDim.x - 1){
            *d_culled_count = offset + local_count;
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < local_count; i += blockDim.x) {
        d_output_gaussians[offset + i] = d_tmp[shared_data[i]];
    }
}
static __global__ void prepareGaussians(Gaussian* d_gaussians,
    int num_gaussians,
    Camera cam,
    int width_stride,
    int height_stride,
    int num_tile_x,
    int num_tile_y,
    int tile_W,
    int tile_H,
    int* threadblock_offsets,
    float k) {
    // threadblock offsets will help us space the lwg

    extern __shared__ int offset_tile[];

    float aspect = cam.aspectRatio;
    float fovY = cam.fovY;
    // k determins the k-sigma radi
    float* R_cam = cam.r_cam;
    float* R_cam_T = cam.r_cam_T;
    float pad = 1.0f; // padding factor
    const float fy = 1.0f / tanf(fovY * 0.5f * (CUDART_PI_F / 180.0f));
    const float fx = fy / aspect;
    int deg = 3;

    float jacobian[6];
    float jacobian_T[6];
    float R[9];
    float R_T[9];
    float scale_mat[9];
    float scale[3];
    float tmp[9];
    float covar[9];
    for (int idx = threadIdx.x; idx < num_tile_x * num_tile_y; idx += blockDim.x) {
        offset_tile[idx] = 0;
    }
    __syncthreads();
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_gaussians; idx += gridDim.x * blockDim.x) {
        Gaussian gauss = d_gaussians[idx];
        // mark invalid by default (xmin > xmax sentinel)
        gauss.aabb[0] = 1; gauss.aabb[1] = 1;
        gauss.aabb[2] = 0; gauss.aabb[3] = 0;
        
        float X, Y, Z;

        X=gauss.X;
        Y=gauss.Y;
        Z=gauss.Z;

        float new_xyz[3];
        new_xyz[0] = gauss.x;
        new_xyz[1] = gauss.y;
        new_xyz[2] = gauss.z;

        // contruct Jacobian
        jacobian[0] = fx / Z; jacobian[1] = 0.0f;
        jacobian[2] = - fx *X / (Z * Z); jacobian[3] = 0.0f;
        jacobian[4] = fy / Z; jacobian[5] = -fy * Y / (Z * Z);

        jacobian_T[0] = jacobian[0]; jacobian_T[1] = jacobian[3]; jacobian_T[2] = jacobian[1];
        jacobian_T[3] = jacobian[4]; jacobian_T[4] = jacobian[2]; jacobian_T[5] = jacobian[5];

        buildRotMatFromQuat_cuda(gauss.rot, R);
        transpose3x3_cuda(R, R_T);
        const float scale_mod = 1.0f;
        scale[0] = scale_mod * gauss.scale[0];
        scale[1] = scale_mod * gauss.scale[1];
        scale[2] = scale_mod * gauss.scale[2];
        buildDiagonalMatrix_cuda(scale, scale_mat);

        // covar = R * S * R_T
        matMul3D_cuda(R, scale_mat, tmp);
        matMul3D_cuda(tmp, scale_mat, R);
        matMul3D_cuda(R, R_T, covar);

        // covar transform to world
        matMul3D_cuda(R_cam, covar, tmp);
        matMul3D_cuda(tmp, R_cam_T, covar);

        // covar  = J * covar * J_T
        geMatMul_cuda(jacobian, covar, 2, 3, 3, tmp);
        float Sigma2D[2*2];
        geMatMul_cuda(tmp, jacobian_T, 2, 2, 3, Sigma2D);
        Sigma2D[0] = (tile_W * 0.5f) * (tile_W * 0.5f) * Sigma2D[0];
        Sigma2D[1] = (tile_W * 0.5f) * (tile_H * 0.5f) * Sigma2D[1];
        Sigma2D[2] = (tile_H * 0.5f) * (tile_W * 0.5f) * Sigma2D[2];
        Sigma2D[3] = (tile_H * 0.5f) * (tile_H * 0.5f) * Sigma2D[3];  

        float invSigma2D[4];
        float det = Sigma2D[0]*Sigma2D[3] - Sigma2D[1]*Sigma2D[2];
        if (det < 1e-8f) {
            d_gaussians[idx] = gauss;
            continue;
        }
        float invDet = 1.0f / det;
        invSigma2D[0] =  Sigma2D[3] * invDet; invSigma2D[1] = -Sigma2D[1] * invDet;
        invSigma2D[2] = -Sigma2D[2] * invDet; invSigma2D[3] =  Sigma2D[0] * invDet;

        gauss.inv_covar[0] = invSigma2D[0];
        gauss.inv_covar[1] = invSigma2D[1];
        gauss.inv_covar[2] = invSigma2D[2];
        gauss.inv_covar[3] = invSigma2D[3];

        // extract eigenvalues
        float Sxx = Sigma2D[0];
        float Sxy = Sigma2D[1];
        float Syx = Sigma2D[2];
        float Syy = Sigma2D[3];

        float sxy = 0.5f*(Sxy + Syx);

        float tr  = Sxx + Syy;
        float dif = Sxx - Syy;
        float rad = sqrtf( fmaxf(0.0f, dif*dif + 4*sxy*sxy) );
        float lamb_1  = 0.5f*(tr + rad);
        float lamb_2  = 0.5f*(tr - rad);
        const float eps = 1e-8f;
        lamb_1 = fmaxf(lamb_1, eps);
        lamb_2 = fmaxf(lamb_2, eps);
        float theta = 0.5f * atan2f(2*sxy, dif); // in radians

        float r1 = k * sqrtf(lamb_1);
        float r2 = k * sqrtf(lamb_2);

        float c = cosf(theta);
        float s = sinf(theta);
        float ex = fabsf(r1*c) + fabsf(r2*s);
        float ey = fabsf(r1*s) + fabsf(r2*c);
        ex /= tile_W/ 2.0f;
        ey /= tile_H/ 2.0f;
        const float max_extent = 2.0f; // in NDC space

        float xmin = new_xyz[0] - ex;
        float xmax = new_xyz[0] + ex;
        float ymin = new_xyz[1] - ey;
        float ymax = new_xyz[1] + ey;

        if (xmax < -0.99f || xmin > 0.99f ||
            ymax < -0.99f || ymin > 0.99f) {
            d_gaussians[idx] = gauss;
            continue;
        }

        xmin = fmaxf(xmin, -1.0f);
        xmax = fminf(xmax, 1.0f);
        ymin = fmaxf(ymin, -1.0f);
        ymax = fminf(ymax, 1.0f);

        int xmin_px = static_cast<int> (floorf(((xmin + 1.0f) * 0.5f) * tile_W));
        int xmax_px = static_cast<int> (ceilf(((xmax + 1.0f) * 0.5f) * tile_W));
        int ymin_px = static_cast<int> (floorf(((ymin + 1.0f) * 0.5f) * tile_H));
        int ymax_px = static_cast<int> (ceilf(((ymax + 1.0f) * 0.5f) * tile_H));

        gauss.px_x = static_cast<int> (roundf(((new_xyz[0] + 1.0f) * 0.5f) * tile_W));
        gauss.px_y = static_cast<int> (roundf(((new_xyz[1] + 1.0f) * 0.5f) * tile_H));

        gauss.aabb[0] = xmin_px;
        gauss.aabb[1] = ymin_px;
        gauss.aabb[2] = xmax_px;
        gauss.aabb[3] = ymax_px;

        int min_x = fmaxf(0, xmin_px / width_stride);
        int max_x = fminf(num_tile_x-1, xmax_px / width_stride);
        int min_y = fmaxf(0, ymin_px / height_stride);
        int max_y = fminf(num_tile_y-1, ymax_px / height_stride);
        

        for (int i = min_x; i <= max_x; i++) {
            for (int j = min_y; j <= max_y; j++) {
                uint32_t tile_id = i + j * num_tile_x;
                atomicAdd(&offset_tile[tile_id], 1);
            }
        }

        d_gaussians[idx] = gauss;
    }

    // __syncthreads();

    // for (int i = threadIdx.x; i < num_tile_x * num_tile_y; i += blockDim.x) {
    //     atomicAdd(&d_tile_info->tile_id_offset[i], offset_tile[i]);
    // }
    __syncthreads();
    for (int i = threadIdx.x; i < num_tile_x * num_tile_y; i+= blockDim.x) {
        threadblock_offsets[(blockIdx.x) * (num_tile_x * num_tile_y) + i] = offset_tile[i];
    }
}

static __global__ void prefixSum(int* start, int stride, int tiles, int log2_tiles) {
    // we have to assume gridDim = 1
    int step = 0;
    int bdx = (int)blockDim.x;
    while (step < log2_tiles){
        int movement = 1 << step;
        for (int i = (tiles-bdx); i >= (movement*stride - bdx + 1); i -= bdx) {
            int idx = threadIdx.x + i;
            int addval = 0;
            if (idx >= movement*stride) {
                addval = start[idx - movement*stride];
            }
            __syncthreads();
            if (idx >= movement*stride) {
                start[idx] += addval;
            }
            __syncthreads();
        }
        step+=1;
        __syncthreads();
    }
}

static __global__ void buildLwgs(Gaussian* d_gaussians,
    lightWeightGaussian* d_lwgs, int num_gaussians,
    Camera cam, int* threadblock_offsets,
    int numBlocks,
    int width_stride,
    int height_stride,
    int num_tile_x,
    int num_tile_y
    ) {
    // this function needs to be used with the SAME number of blocks/threads as prepareGaussians
    extern __shared__ int offset_tile[]; // so for just idx it will store the local indices

    for (int idx = threadIdx.x; idx < num_tile_x * num_tile_y; idx += blockDim.x) {
        offset_tile[idx] = 0;
    }
    __syncthreads();
    lightWeightGaussian lwg;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_gaussians; idx += gridDim.x * blockDim.x) {
        Gaussian gauss = d_gaussians[idx];
        if (gauss.aabb[0] > gauss.aabb[2] || gauss.aabb[1] > gauss.aabb[3]) continue;
        int xmin_px = gauss.aabb[0];
        int ymin_px = gauss.aabb[1];
        int xmax_px = gauss.aabb[2];
        int ymax_px = gauss.aabb[3];

        int min_x = fmaxf(0, xmin_px / width_stride);
        int max_x = fminf(num_tile_x-1, xmax_px / width_stride);
        int min_y = fmaxf(0, ymin_px / height_stride);
        int max_y = fminf(num_tile_y-1, ymax_px / height_stride);
        for (int i = min_x; i <= max_x; i++) {
            for (int j = min_y; j <= max_y; j++) {
                int tile_idx = i + j * num_tile_x;
                int cur_idx = atomicAdd(&offset_tile[tile_idx], 1);
                int threadb_offset = blockIdx.x == 0 ? 0 : threadblock_offsets[(blockIdx.x - 1) * (num_tile_x * num_tile_y) + tile_idx];
                int glob_offset = tile_idx == 0 ? 0 : threadblock_offsets[(numBlocks-1) * (num_tile_x * num_tile_y) + tile_idx-1];
                int global_index = cur_idx + glob_offset + threadb_offset;

                lwg.gaussian_id = static_cast<uint32_t> (idx);
                lwg.radix_id = (static_cast<uint64_t>(tile_idx) << 32) | static_cast<uint32_t> (-gauss.Z * 1e6f);
                d_lwgs[global_index] = lwg;
                // lwg.radix_id = static_cast<uint64_t> (gauss.z * 1e6f);
            }
        }

    }
}

static int ceil_log2(int n) {
    int k = 0;
    int v = 1;
    while (v < n) { v <<= 1; ++k; }
    return k;
}

static __global__ void extractKeys(const lightWeightGaussian* in, uint64_t* keys, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) keys[idx] = in[idx].radix_id;
}

extern "C" void preprocessCUDAGaussians(Gaussian* d_gaussians,
    float * out_pixels,
    int num_gaussians,
    Camera cam,
    int num_tile_y,
    int num_tile_x,
    int width_stride,
    int height_stride,
    int tile_W,
    int tile_H,
    float k) {
    
    int BLOCK_SIZE = 256;
    int NUM_BLOCKS = 128;

    int width = BLOCK_SIZE * NUM_BLOCKS;
    int stride = (num_gaussians + width - 1) / width;
    
    // printf("preprocessCUDAGaussians: num_gaussians before cull = %d\n", num_gaussians);

    Gaussian* d_culled_gaussians; 
    cudaMalloc(&d_culled_gaussians, sizeof(Gaussian) * num_gaussians); // worst case all gaussians pass culling

    int* d_culled_count;
    cudaMalloc(&d_culled_count, sizeof(int));
    cudaMemset(d_culled_count, 0, sizeof(int));

    int* d_threadblock_counts;
    cudaMalloc(&d_threadblock_counts, sizeof(int) * NUM_BLOCKS);
    cudaMemset(d_threadblock_counts, 0, sizeof(int) * NUM_BLOCKS);
    Gaussian* d_tmp;
    cudaMalloc(&d_tmp, sizeof(Gaussian) * num_gaussians);

    size_t cull_gaussian_shared_mem = sizeof(int) * BLOCK_SIZE * stride;
    advancedCullGaussians<<<NUM_BLOCKS, BLOCK_SIZE, cull_gaussian_shared_mem>>>(
        d_gaussians,
        d_culled_gaussians,
        d_tmp,
        num_gaussians,
        cam,
        d_culled_count,
        d_threadblock_counts
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cullGaussians launch error: %s\n", cudaGetErrorString(err));
    }

    // Then check execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("advancedCullGaussians runtime error: %s\n", cudaGetErrorString(err));
    }
    int h_culled_count = 0;
    cudaMemcpy(&h_culled_count, d_culled_count, sizeof(int), cudaMemcpyDeviceToHost);

    // printf("preprocessCUDAGaussians: num_gaussians after cull = %d\n", h_culled_count);

    size_t prepare_gaussian_shared_mem = (num_tile_x * num_tile_y) * sizeof(int);
    cudaFree(d_threadblock_counts);
    cudaMalloc(&d_threadblock_counts, sizeof(int) * NUM_BLOCKS * (num_tile_x * num_tile_y ));
    cudaMemset(d_threadblock_counts, 0, sizeof(int) * NUM_BLOCKS * (num_tile_x * num_tile_y ));

    prepareGaussians<<<NUM_BLOCKS, BLOCK_SIZE, prepare_gaussian_shared_mem>>>(d_culled_gaussians, h_culled_count, cam, 
        width_stride, height_stride, num_tile_x, num_tile_y, tile_W, tile_H, d_threadblock_counts, 2.0f);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("prepareGaussians launch error: %s\n", cudaGetErrorString(err));
    }

    // Then check execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("prepareGaussians runtime error: %s\n", cudaGetErrorString(err));
    }

    int log2_blocks = ceil_log2(NUM_BLOCKS);
    int log2_tiles = ceil_log2(num_tile_x * num_tile_y);
    prefixSum<<<1, 512>>>(d_threadblock_counts, num_tile_y * num_tile_x,  NUM_BLOCKS * (num_tile_x * num_tile_y), log2_blocks);
    prefixSum<<<1, 512>>>(d_threadblock_counts + (NUM_BLOCKS - 1) * (num_tile_x * num_tile_y), 1, (num_tile_x * num_tile_y), log2_tiles);
    // prefixSum<<<1, 256>>>(d_offsets, 1, tile_info->num_tile_x * tile_info->num_tile_y, log2_tiles);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("prepareGaussians launch error: %s\n", cudaGetErrorString(err));
    }

    // Then check execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("prepareGaussians runtime error: %s\n", cudaGetErrorString(err));
    }
    int total_count;
    cudaMemcpy(&total_count, &d_threadblock_counts[NUM_BLOCKS * (num_tile_x * num_tile_y) - 1], sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << "Total count of culled gaussians: " << total_count << std::endl;
    // print culled_count
    lightWeightGaussian* d_lwgs;
    cudaMalloc(&d_lwgs, sizeof(lightWeightGaussian) * total_count);
    int * tile_offsets;
    cudaMalloc(&tile_offsets, sizeof(int) * (num_tile_x * num_tile_y));
    cudaMemcpy(tile_offsets, d_threadblock_counts + (NUM_BLOCKS - 1) * (num_tile_x * num_tile_y), sizeof(int) * (num_tile_x * num_tile_y), cudaMemcpyDeviceToDevice);
    size_t build_lwg_shared_mem = (num_tile_x *num_tile_y) * sizeof(int);
    buildLwgs<<<NUM_BLOCKS, BLOCK_SIZE, build_lwg_shared_mem>>>(
        d_culled_gaussians,
        d_lwgs,
        h_culled_count,
        cam,
        d_threadblock_counts,
        NUM_BLOCKS,
        width_stride,
        height_stride,
        num_tile_x,
        num_tile_y
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("build lwgs launch error: %s\n", cudaGetErrorString(err));
    }

    // Then check execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("build lwgs runtime error: %s\n", cudaGetErrorString(err));
    }
    // int num_bits = 48;
    // int N = total_count;
    // int* d_global_counter;
    // int* d_lookback;
    // lightWeightGaussian* d_input;
    // lightWeightGaussian* d_output;
    // cudaMalloc(&d_output, N * sizeof(lightWeightGaussian));
    // int numPasses = (num_bits + 7) / 8;

    // BLOCK_SIZE = 256;
    // NUM_BLOCKS = 174;
    
    // int zero = 0;

    // int TOTAL_TILES = (N + TILE_SIZE - 1) / TILE_SIZE;
    // size_t SHARED_MEMORY_SIZE = RADIX * numPasses * (BLOCK_SIZE/32) * sizeof(int);
    // // std::cout << "Shared mem size: " << SHARED_MEMORY_SIZE / 1024 << " KB" << std::endl;

    // cudaMalloc(&d_global_counter, numPasses * (RADIX ) * sizeof(int));
    // cudaMalloc(&d_lookback, TOTAL_TILES * (RADIX) * sizeof(int));
    // cudaMalloc(&d_input, N * sizeof(lightWeightGaussian));
    // cudaMemset(d_global_counter, 0, numPasses * (RADIX) * sizeof(int));

    // cudaMemcpy(d_input, d_lwgs, N * sizeof(lightWeightGaussian), cudaMemcpyDeviceToDevice);
    // cudaMemcpy(d_output, d_lwgs, N * sizeof(lightWeightGaussian), cudaMemcpyDeviceToDevice);

    // lightWeightGaussian* in = d_input;
    // lightWeightGaussian* out = d_output;

    // globalBinCounter<<<NUM_BLOCKS, BLOCK_SIZE, SHARED_MEMORY_SIZE>>>(d_input, d_global_counter, numPasses, N);
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("global bin  launch error: %s\n", cudaGetErrorString(err));
    // }

    // // Then check execution errors
    // err = cudaDeviceSynchronize();
    // if (err != cudaSuccess) {
    //     printf("global bin  runtime error: %s\n", cudaGetErrorString(err));
    // }
    // for (int shift = 0; shift < numPasses; ++shift) {
    //     // cudaMemcpyToSymbol(gTileCounter, &zero, sizeof(int));
    //     cudaMemset(d_lookback, 0, TOTAL_TILES * (RADIX) * sizeof(int));
    //     oneSweep<<<TOTAL_TILES, BLOCK_SIZE>>>(in, out, d_lookback, d_global_counter, N, shift);
    //     std::swap(in, out);
    //     // cudaMemcpy(d_input, d_output, N * sizeof(int), cudaMemcpyDeviceToDevice);
    // }

    // lightWeightGaussian* result = (numPasses & 1) ? in : out; 
    // cudaMemcpy(d_lwgs, result, N * sizeof(lightWeightGaussian), cudaMemcpyDeviceToDevice);
    // float * d_out;
    // cudaMalloc(&d_out, sizeof(float) * 3 * tile_H * tile_W);
    // cudaMemset(d_out, 0, sizeof(float) * 3 * tile_H * tile_W);

    // BLOCK_SIZE = 256;
    // NUM_BLOCKS = 256;
    
    // size_t shared_mem_size = (height_stride * width_stride) * (1 + 3) * sizeof(float);
    // renderGaussians<<<NUM_BLOCKS, BLOCK_SIZE, shared_mem_size>>>(d_out, tile_offsets, d_culled_gaussians, d_lwgs,
    //                                         height_stride, width_stride, tile_W, tile_H, num_tile_x, num_tile_y);
    // cudaMemcpy(out_pixels, d_out, sizeof(float) * 3 * tile_H * tile_W, cudaMemcpyDeviceToHost);
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("one sweep launch error: %s\n", cudaGetErrorString(err));
    // }

    // // Then check execution errors
    // err = cudaDeviceSynchronize();
    // if (err != cudaSuccess) {
    //     printf("one sweep runtime error: %s\n", cudaGetErrorString(err));
    // }
    

    // cudaFree(tile_offsets);
    // cudaFree(d_tmp);
    // cudaFree(d_global_counter);
    // cudaFree(d_input);
    // cudaFree(d_output);
    // cudaFree(d_lookback);
    // cudaFree(d_out);
    // cudaFree(d_culled_gaussians);
    // cudaFree(d_culled_count);
    // cudaFree(d_threadblock_counts);
    // cudaFree(d_offsets);
    // cudaFree(d_lwgs);

    int N = total_count;

    uint64_t* d_keys_in  = nullptr;
    uint64_t* d_keys_out = nullptr;
    lightWeightGaussian* d_lwgs_sorted = nullptr;

    cudaMalloc(&d_keys_in,  N * sizeof(uint64_t));
    cudaMalloc(&d_keys_out, N * sizeof(uint64_t));
    cudaMalloc(&d_lwgs_sorted, N * sizeof(lightWeightGaussian));

    // keys = radix_id
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    extractKeys<<<blocks, threads>>>(d_lwgs, d_keys_in, N);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("extractKeys runtime error: %s\n", cudaGetErrorString(err));
    }

    // CUB temp storage query + run
    void*  d_temp_storage = nullptr;
    size_t temp_bytes = 0;

    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes,
        d_keys_in, d_keys_out,
        d_lwgs, d_lwgs_sorted,
        N,
        /*begin_bit=*/0, /*end_bit=*/64
    );
    cudaMalloc(&d_temp_storage, temp_bytes);

    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_bytes,
        d_keys_in, d_keys_out,
        d_lwgs, d_lwgs_sorted,
        N,
        0, 64
    );
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUB sort runtime error: %s\n", cudaGetErrorString(err));
    }

    // Use d_lwgs_sorted for rendering (no need to copy back unless you want to)
    float* d_out = nullptr;
    cudaMalloc(&d_out, sizeof(float) * 3 * tile_H * tile_W);
    cudaMemset(d_out, 0, sizeof(float) * 3 * tile_H * tile_W);

    BLOCK_SIZE = 256;
    NUM_BLOCKS = 256;
    size_t shared_mem_size = (height_stride * width_stride) * (1 + 3) * sizeof(float);

    renderGaussians<<<NUM_BLOCKS, BLOCK_SIZE, shared_mem_size>>>(
        d_out, tile_offsets, d_culled_gaussians, d_lwgs_sorted,
        height_stride, width_stride, tile_W, tile_H, num_tile_x, num_tile_y
    );
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("renderGaussians runtime error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(out_pixels, d_out, sizeof(float) * 3 * tile_H * tile_W, cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(d_out);
    cudaFree(d_temp_storage);
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_lwgs_sorted);

    cudaFree(tile_offsets);
    cudaFree(d_tmp);
    cudaFree(d_culled_gaussians);
    cudaFree(d_culled_count);
    cudaFree(d_threadblock_counts);
    cudaFree(d_lwgs);
}
