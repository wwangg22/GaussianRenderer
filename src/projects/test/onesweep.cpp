// correctness_onesweep_vs_cub.cpp
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// ---- External kernels you provide elsewhere ----
extern "C" void oneSweepSort  (int *input_array, int *output_array, int N, int max_val, float* kernel_ms);
extern "C" void launchCUBRadixSortPairs(const int* h_in, int* h_keys_out, int* h_idx_out, int N, int maxVal, float* sort_ms);

// ---- CUDA error helper ----
static inline void cudaCheck(cudaError_t e, const char* what, const char* file, int line) {
    if (e != cudaSuccess) {
        std::cerr << "[CUDA] " << what << " failed at " << file << ":" << line
                  << " → " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}
#define CUDA_OK(cmd) cudaCheck((cmd), #cmd, __FILE__, __LINE__)

// ---- CLI parsing (minimal) ----
struct Args {
    int minN   = 2048;           // default: start small and consecutive
    int maxN   = 4096;           // default: inclusive upper bound
    double factor = 1.15;        // used only with --geometric
    int reps   = 3;
    int seeds  = 1;
    int maxVal = (1 << 24) - 1;
    bool stop_on_fail = false;
    bool geometric = false;      // default to consecutive sizes
};

static inline bool starts_with(const std::string& s, const std::string& p){ return s.rfind(p, 0) == 0; }

Args parse_args(int argc, char** argv){
    Args a;
    for (int i=1;i<argc;++i){
        std::string tok = argv[i];
        if (tok=="--minN"   && i+1<argc) { a.minN   = std::atoi(argv[++i]); }
        else if (tok=="--maxN"   && i+1<argc) { a.maxN   = std::atoi(argv[++i]); }
        else if (tok=="--factor" && i+1<argc) { a.factor = std::atof(argv[++i]); }
        else if (tok=="--reps"   && i+1<argc) { a.reps   = std::atoi(argv[++i]); }
        else if (tok=="--seeds"  && i+1<argc) { a.seeds  = std::atoi(argv[++i]); }
        else if (tok=="--maxVal" && i+1<argc) { a.maxVal = std::atoi(argv[++i]); }
        else if (tok=="--stop_on_fail") { a.stop_on_fail = true; }
        else if (tok=="--geometric") { a.geometric = true; }
        else if (starts_with(tok, "--help")) {
            std::cout <<
            "Usage: ./correctness [--minN 2048] [--maxN 4096]\n"
            "                     [--geometric] [--factor 1.15]\n"
            "                     [--reps 3] [--seeds 1] [--maxVal 16777215]\n"
            "                     [--stop_on_fail]\n";
            std::exit(0);
        }
    }
    const int LIMIT = (1u<<30) - 1;
    if (a.maxN > LIMIT) a.maxN = LIMIT;
    if (a.minN < 1) a.minN = 1;
    if (a.factor < 1.01) a.factor = 1.01;
    if (a.maxN < a.minN) a.maxN = a.minN;  // guard
    return a;
}

// ---- Helpers ----
// Geometric generator (original behavior)
static inline std::vector<int> make_sizes_geometric(int minN, int maxN, double factor) {
    std::vector<int> sizes;
    if (minN < 1) minN = 1;
    if (maxN < minN) maxN = minN;
    double cur = static_cast<double>(minN);
    while (cur < static_cast<double>(maxN)) {
        sizes.push_back(static_cast<int>(cur));
        cur = cur * factor;
        if (static_cast<int>(cur) <= sizes.back()) cur = sizes.back() + 1.0; // guard
    }
    const int LIMIT = (1u<<30) - 1; // < 2^30
    for (int &v : sizes) if (v > LIMIT) v = LIMIT;
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    sizes.erase(std::remove_if(sizes.begin(), sizes.end(), [](int x){return x<=0;}), sizes.end());
    if (sizes.empty() || sizes.back() != maxN) sizes.push_back(maxN);
    return sizes;
}

// Consecutive sizes generator [minN, maxN] inclusive
static inline std::vector<int> make_sizes_consecutive(int minN, int maxN) {
    std::vector<int> sizes;
    sizes.reserve(std::max(0, maxN - minN + 1));
    for (int n = minN; n <= maxN; ++n) sizes.push_back(n);
    return sizes;
}

static inline bool is_nondecreasing(const std::vector<int>& v){
    for (size_t i=1;i<v.size();++i) if (v[i-1] > v[i]) return false;
    return true;
}

static void print_first_mismatches(const std::vector<int>& got,
                                   const std::vector<int>& ref,
                                   int max_show = 10)
{
    int shown = 0;
    for (size_t i=0; i<got.size() && i<ref.size(); ++i) {
        if (got[i] != ref[i]) {
            if (shown == 0) std::cerr << "First mismatches (index: got != ref):\n";
            std::cerr << "  [" << i << "]: " << got[i] << " != " << ref[i] << "\n";
            if (++shown >= max_show) break;
        }
    }
    if (shown == 0) std::cerr << "(No element-by-element mismatches found within compared range)\n";
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    // CUDA device info
    CUDA_OK(cudaSetDevice(0));
    cudaDeviceProp p{}; CUDA_OK(cudaGetDeviceProperties(&p, 0));
    std::cout << "GPU: " << p.name << " (SM " << p.major << "." << p.minor << ")\n";
    std::cout << "Correctness check for OneSweep vs CUB (keys)\n";
    std::cout << "Sizes: minN=" << args.minN << " maxN=" << args.maxN
              << " mode=" << (args.geometric ? "geometric" : "consecutive")
              << (args.geometric ? (", factor=" + std::to_string(args.factor)) : "")
              << "\n";
    std::cout << "reps=" << args.reps << " seeds=" << args.seeds
              << " maxVal=" << args.maxVal
              << (args.stop_on_fail ? " [stop_on_fail]\n" : "\n");

    // Warm-up OneSweep (tiny)
    {
        std::vector<int> small_in(4096), small_out(4096);
        for (int i = 0; i < 4096; ++i) small_in[i] = i & 2500;
        float dummy = 0.0f;
        oneSweepSort(small_in.data(), small_out.data(), 4096, 2500, &dummy);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
    }

    // Build sizes list
    auto sizes = args.geometric
        ? make_sizes_geometric(args.minN, args.maxN, args.factor)
        : make_sizes_consecutive(args.minN, args.maxN);

    int total_tests = 0;
    int total_pass  = 0;

    for (int N : sizes) {
        std::cout << "\n=== N = " << N << " ===\n";
        for (int si = 0; si < args.seeds; ++si) {
            uint64_t seed = 12345 + si;
            std::mt19937_64 rng(seed);
            std::uniform_int_distribution<int> dist(0, args.maxVal);

            // Base input (same for all reps to make diffs meaningful per seed)
            std::vector<int> base(N);
            for (int i = 0; i < N; ++i) base[i] = dist(rng);

            for (int r = 0; r < args.reps; ++r) {
                ++total_tests;

                // Prepare host buffers
                std::vector<int> h_in = base;
                std::vector<int> h_cub(N), h_idx(N), h_one(N);

                // CUB pairs → reference keys
                {
                    float dummy = 0.0f;
                    launchCUBRadixSortPairs(h_in.data(), h_cub.data(), h_idx.data(), N, args.maxVal, &dummy);
                    CUDA_OK(cudaGetLastError());
                    CUDA_OK(cudaDeviceSynchronize());
                }

                // OneSweep (keys-only)
                {
                    float dummy = 0.0f;
                    oneSweepSort(h_in.data(), h_one.data(), N, args.maxVal, &dummy);
                    CUDA_OK(cudaGetLastError());
                    CUDA_OK(cudaDeviceSynchronize());
                }

                const bool nondec  = is_nondecreasing(h_one);
                const bool size_ok = (h_one.size() == h_cub.size());
                bool keys_ok = size_ok &&
                               std::equal(h_one.begin(), h_one.end(), h_cub.begin());

                const bool pass = (nondec && size_ok && keys_ok);

                if (pass) {
                    ++total_pass;
                    std::cout << "Seed " << seed << " r=" << r << " : PASS\n";
                } else {
                    std::cout << "Seed " << seed << " r=" << r << " : FAIL\n";
                    if (!nondec) std::cerr << " - Output is NOT non-decreasing\n";
                    if (!size_ok) std::cerr << " - Size mismatch: one=" << h_one.size()
                                            << " cub=" << h_cub.size() << "\n";
                    if (!keys_ok && size_ok) {
                        print_first_mismatches(h_one, h_cub, 10);
                    }
                    if (args.stop_on_fail) {
                        std::cerr << "\nStopping on first failure (use without --stop_on_fail to continue).\n";
                        std::cerr << "Summary: " << total_pass << "/" << total_tests << " passed.\n";
                        return 1;
                    }
                }
            } // reps
        } // seeds
    } // sizes

    std::cout << "\nSummary: " << total_pass << " / " << total_tests << " tests passed.\n";
    return (total_pass == total_tests) ? 0 : 1;
}
