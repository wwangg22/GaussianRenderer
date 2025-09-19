// benchmark_radix_sort.cpp
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>   // for std::pair
#include <vector>

extern "C" void launchRadixSort(int *input_array, int *output_array, int N, int max_val);
extern "C" void launchRadixSort_4(int *input_array, int *output_array, int N, int max_val);
extern "C" void oneSweepSort(int *input_array, int *output_array, int N, int max_val, float* kernel_ms);

// NEW: CUB wrapper (implemented in cub_sort.cu)
extern "C" void launchCUBRadixSortPairs(const int* h_in,
                                        int* h_keys_out,
                                        int* h_idx_out,
                                        int N,
                                        int maxVal,
                                        float* sort_ms);

static inline void cudaCheck(cudaError_t e, const char* what, const char* file, int line) {
    if (e != cudaSuccess) {
        std::cerr << "[CUDA] " << what << " failed at " << file << ":" << line
                  << " → " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}
#define CUDA_OK(cmd) cudaCheck((cmd), #cmd, __FILE__, __LINE__)

int main(int argc, char** argv) {
    // Defaults
    int dev = 0;
    int N = 1 << 22;                 // 4M keys
    int threadsPerBlock = 256;       // your kernel expects >= RADIX(=16)
    int maxVal = (1 << 24) - 1;      // keys in [0, maxVal]
    uint64_t seed = 12345;

    // CLI: N threadsPerBlock maxVal seed
    if (argc >= 2) N = std::atoi(argv[1]);
    if (argc >= 3) threadsPerBlock = std::atoi(argv[2]);
    if (argc >= 4) maxVal = std::atoi(argv[3]);
    if (argc >= 5) seed = static_cast<uint64_t>(std::strtoull(argv[4], nullptr, 10));

    CUDA_OK(cudaSetDevice(dev));
    cudaDeviceProp p{}; CUDA_OK(cudaGetDeviceProperties(&p, dev));
    std::cout << "GPU: " << p.name << " (SM " << p.major << "." << p.minor << ")\n";
    std::cout << "N=" << N << ", threadsPerBlock=" << threadsPerBlock
              << ", maxVal=" << maxVal << ", seed=" << seed << "\n";

    // Generate input with duplicates (to check stability)
    std::vector<int> h_in(N), h_out(N);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dist(0, maxVal);
    for (int i = 0; i < N; ++i) h_in[i] = dist(rng);

    // Golden: stable sort reference (host) using pairs (key, original_index)
    std::vector<std::pair<int,int>> ref(N);
    for (int i = 0; i < N; ++i) ref[i] = {h_in[i], i};
    auto t0_ref = std::chrono::high_resolution_clock::now();
    std::stable_sort(ref.begin(), ref.end(),
        [](const auto& a, const auto& b){ return a.first < b.first; });
    auto t1_ref = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> ref_sec = t1_ref - t0_ref;

    // Split ref into keys/indices for checks
    std::vector<int> ref_keys(N), ref_idx(N);
    for (int i = 0; i < N; ++i) { ref_keys[i] = ref[i].first; ref_idx[i] = ref[i].second; }

    // Warm-up (small) to create context
    {
        float test_ms = 0.0f;
        std::vector<int> small_in(4096), small_out(4096);
        for (int i = 0; i < 4096; ++i) small_in[i] = i & 2500;
        oneSweepSort(small_in.data(), small_out.data(), 4096, 2500, &test_ms);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());

         // CPU reference sort
        std::vector<int> ref_small = small_in;
        std::stable_sort(ref_small.begin(), ref_small.end());

        // Check correctness
       auto first_diff = std::mismatch(small_out.begin(), small_out.end(), ref_small.begin());
    bool ok = (first_diff.first == small_out.end());
    if (!ok) {
        const size_t N = small_out.size();
        size_t i0 = size_t(std::distance(small_out.begin(), first_diff.first));
        std::cerr << "[Warm-up] Mismatch detected! N=" << N << "\n";
        std::cerr << "First mismatch at i=" << i0
                  << "  GPU=" << *first_diff.first
                  << "  REF=" << *first_diff.second << "\n";

        // Show up to 10 mismatches with local context
        int shown = 0;
        for (size_t i = 0; i < N && shown < 10; ++i) {
            if (small_out[i] != ref_small[i]) {
                int prev = (i ? small_out[i-1] : std::numeric_limits<int>::min());
                int next = (i+1<N ? small_out[i+1] : std::numeric_limits<int>::max());
                std::cerr << "  i=" << i
                          << "  GPU=" << small_out[i] << " (prev=" << prev << ", next=" << next << ")"
                          << "  REF=" << ref_small[i] << "\n";
                ++shown;
            }
        }

        // Inversion (sortedness) check on GPU output
        size_t invs = 0;
        for (size_t i = 1; i < N && invs < 10; ++i) {
            if (small_out[i-1] > small_out[i]) {
                std::cerr << "  inversion at (" << (i-1) << "," << i
                          << "): " << small_out[i-1] << " > " << small_out[i] << "\n";
                ++invs;
            }
        }
        if (invs == 0) std::cerr << "  (GPU output appears non-decreasing)\n";

        // Lightweight permutation checks (sums + xor)
        auto quick_sig = [](const std::vector<int>& v){
            unsigned long long s=0, s2=0, x=0;
            for (int a : v){ s+= (unsigned long long)a; s2+= (unsigned long long)a*(unsigned long long)a; x^= (unsigned long long)a*11400714819323198485ull; }
            return std::tuple<unsigned long long,unsigned long long,unsigned long long>(s,s2,x);
        };
        auto [sg, s2g, xg] = quick_sig(small_out);
        auto [sr, s2r, xr] = quick_sig(ref_small);
        std::cerr << "Checksums (GPU vs REF): sum " << sg << " vs " << sr
                  << " | sum2 " << s2g << " vs " << s2r
                  << " | xor "  << xg  << " vs " << xr  << "\n";

        // Keep your existing quick peek
        std::cerr << "GPU out (first 32): ";
        for (size_t i = 0; i < std::min<size_t>(32, N); ++i) std::cerr << small_out[i] << (i+1==std::min<size_t>(32,N)?'\n':' ');
        std::cerr << "CPU ref (first 32): ";
        for (size_t i = 0; i < std::min<size_t>(32, N); ++i) std::cerr << ref_small[i] << (i+1==std::min<size_t>(32,N)?'\n':' ');

        std::exit(1);
        } else {
            std::cout << "[Warm-up] GPU sort matches CPU reference.\n";
        }
    }
    // ---- Your GPU sort (keys-only) ----
    float one_sweep_sort_ms = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();
    oneSweepSort(h_in.data(), h_out.data(), N, maxVal, &one_sweep_sort_ms);

    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_sec = t1 - t0;

    // Correctness for your GPU sort
    bool nondecreasing = true;
    for (int i = 1; i < N; ++i) {
        if (h_out[i-1] > h_out[i]) { nondecreasing = false; break; }
    }
    bool stable_ok = std::equal(h_out.begin(), h_out.end(), ref_keys.begin());

    // ---- CUB radix sort (pairs) via wrapper ----
    std::vector<int> h_out_cub(N), h_idx_cub(N);
    float cub_sort_ms = 0.0f; // device sort-only time from events inside wrapper
    auto t0_cub_wall = std::chrono::high_resolution_clock::now();
    launchCUBRadixSortPairs(h_in.data(), h_out_cub.data(), h_idx_cub.data(), N, maxVal, &cub_sort_ms);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    auto t1_cub_wall = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cub_wall = t1_cub_wall - t0_cub_wall;

    bool nondecreasing_cub = true;
    for (int i = 1; i < N; ++i) {
        if (h_out_cub[i-1] > h_out_cub[i]) { nondecreasing_cub = false; break; }
    }
    bool keys_match_ref_cub = std::equal(h_out_cub.begin(), h_out_cub.end(), ref_keys.begin());
    bool idx_match_ref_cub  = std::equal(h_idx_cub.begin(),  h_idx_cub.end(),  ref_idx.begin());

    // Report
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CPU stable_sort         : " << (ref_sec.count()*1000.0)   << " ms\n";
    std::cout << "GPU radix sort (yours)  : " << one_sweep_sort_ms << " ms device, "
              << (gpu_sec.count()*1000.0)   << " ms wall  "
              << "(" << (N / (one_sweep_sort_ms/1000.0) / 1e6) << " Mkeys/s)\n";
    std::cout << "CUB radix sort (pairs)  : " << cub_sort_ms << " ms device, "
              << (cub_wall.count()*1000.0) << " ms wall  "
              << "(" << (N / (cub_sort_ms/1000.0) / 1e6) << " Mkeys/s)\n";

    std::cout << "Your sort correctness   : nondecreasing="
              << (nondecreasing ? "OK" : "FAIL")
              << ", keys==ref=" << (stable_ok ? "OK" : "FAIL") << "\n";

    std::cout << "CUB sort correctness    : nondecreasing="
              << (nondecreasing_cub ? "OK" : "FAIL")
              << ", keys==ref=" << (keys_match_ref_cub ? "OK" : "FAIL")
              << ", indices(stability)==ref=" << (idx_match_ref_cub ? "OK" : "FAIL") << "\n";

    int rc_custom = (nondecreasing && stable_ok) ? 0 : 2;
    int rc_cub    = (nondecreasing_cub && keys_match_ref_cub && idx_match_ref_cub) ? 0 : 3;

    if (rc_custom != 0) {
        std::cerr << "[Your sort] Example (first 16 output keys): ";
        for (int i = 0; i < std::min(N,16); ++i) std::cerr << h_out[i] << " ";
        std::cerr << "\n[Ref keys ] Example (first 16): ";
        for (int i = 0; i < std::min(N,16); ++i) std::cerr << ref_keys[i] << " ";
        std::cerr << "\n";
    }
    if (rc_cub != 0) {
        std::cerr << "[CUB keys] Example (first 16): ";
        for (int i = 0; i < std::min(N,16); ++i) std::cerr << h_out_cub[i] << " ";
        std::cerr << "\n[Ref keys] Example (first 16): ";
        for (int i = 0; i < std::min(N,16); ++i) std::cerr << ref_keys[i] << " ";
        std::cerr << "\n[CUB idx ] Example (first 16): ";
        for (int i = 0; i < std::min(N,16); ++i) std::cerr << h_idx_cub[i] << " ";
        std::cerr << "\n[Ref idx ] Example (first 16): ";
        for (int i = 0; i < std::min(N,16); ++i) std::cerr << ref_idx[i] << " ";
        std::cerr << "\n";
    }

    return (rc_custom == 0 && rc_cub == 0) ? 0 : 1;
}
