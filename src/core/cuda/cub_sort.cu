// cub_sort.cu
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdint>

static inline void cudaCheck(cudaError_t e, const char* what, const char* file, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "[CUDA] %s failed at %s:%d → %s\n", what, file, line, cudaGetErrorString(e));
        abort();
    }
}
#define CUDA_OK(cmd) cudaCheck((cmd), #cmd, __FILE__, __LINE__)

static inline int bits_needed(uint32_t max_val) {
    if (max_val == 0) return 1;
    int b = 0;
    while (max_val) { ++b; max_val >>= 1; }
    return b;
}

// Host-callable wrapper (compiled with NVCC)
// - h_in       : host input keys (size N)
// - h_keys_out : host output sorted keys (size N)
// - h_idx_out  : host output “stable order” original indices (size N)
// - sort_ms    : if non-null, returns device sort-only time in milliseconds
extern "C" void launchCUBRadixSortPairs(const int* h_in,
                                        int* h_keys_out,
                                        int* h_idx_out,
                                        int N,
                                        int maxVal,
                                        float* sort_ms)
{
    if (N <= 0) { if (sort_ms) *sort_ms = 0.0f; return; }

    const int begin_bit = 0;
    const int end_bit   = bits_needed(static_cast<uint32_t>(maxVal)); // or 8*sizeof(int)

    // Device buffers
    int *d_keys_in=nullptr, *d_keys_out=nullptr;
    int *d_vals_in=nullptr, *d_vals_out=nullptr;
    CUDA_OK(cudaMalloc(&d_keys_in,  N * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_keys_out, N * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_vals_in,  N * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_vals_out, N * sizeof(int)));

    // Host indices [0..N-1]
    int* h_idx = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; ++i) h_idx[i] = i;

    // H2D
    CUDA_OK(cudaMemcpy(d_keys_in, h_in,   N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_vals_in, h_idx,  N * sizeof(int), cudaMemcpyHostToDevice));

    // CUB temp storage
    void*  d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes,
        d_keys_in, d_keys_out,
        d_vals_in, d_vals_out,
        N, begin_bit, end_bit
    );
    CUDA_OK(cudaMalloc(&d_temp, temp_bytes));

    // Time just the sort with CUDA events (device-side duration)
    cudaEvent_t ev_start, ev_stop;
    CUDA_OK(cudaEventCreate(&ev_start));
    CUDA_OK(cudaEventCreate(&ev_stop));
    CUDA_OK(cudaEventRecord(ev_start));
    cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes,
        d_keys_in, d_keys_out,
        d_vals_in, d_vals_out,
        N, begin_bit, end_bit
    );
    CUDA_OK(cudaEventRecord(ev_stop));
    CUDA_OK(cudaEventSynchronize(ev_stop));

    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    if (sort_ms) *sort_ms = ms;

    // D2H
    CUDA_OK(cudaMemcpy(h_keys_out, d_keys_out, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_idx_out,  d_vals_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_OK(cudaEventDestroy(ev_start));
    CUDA_OK(cudaEventDestroy(ev_stop));
    CUDA_OK(cudaFree(d_temp));
    CUDA_OK(cudaFree(d_keys_in));
    CUDA_OK(cudaFree(d_keys_out));
    CUDA_OK(cudaFree(d_vals_in));
    CUDA_OK(cudaFree(d_vals_out));
    free(h_idx);
}
