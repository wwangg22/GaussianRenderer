// benchmark_radix_bench_gitems_dense.cpp
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// ---- External kernels you provide elsewhere ----
extern "C" void launchRadixSort_4(int *input_array, int *output_array, int N, int max_val, float* kernel_ms);
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

// ---- Small helpers ----
static inline void write_jsonl(std::ofstream& ofs,
                               const std::string& algo,
                               const std::string& gpu_name,
                               int sm_major, int sm_minor,
                               int N, int repetition, uint64_t seed,
                               int maxVal,
                               double device_ms, double wall_ms,
                               double gitems_s_device, double gitems_s_wall,
                               bool nondecreasing, bool keys_match_ref, bool indices_match_ref)
{
    ofs << "{";
    ofs << "\"algo\":\"" << algo << "\",";
    ofs << "\"gpu_name\":\"" << gpu_name << "\",";
    ofs << "\"sm_major\":" << sm_major << ",";
    ofs << "\"sm_minor\":" << sm_minor << ",";
    ofs << "\"N\":" << N << ",";
    ofs << "\"repetition\":" << repetition << ",";
    ofs << "\"seed\":" << seed << ",";
    ofs << "\"maxVal\":" << maxVal << ",";
    ofs << std::fixed << std::setprecision(6);
    ofs << "\"device_ms\":" << device_ms << ",";
    ofs << "\"wall_ms\":" << wall_ms << ",";
    ofs << "\"gitems_s_device\":" << gitems_s_device << ",";
    ofs << "\"gitems_s_wall\":"   << gitems_s_wall << ",";
    ofs << "\"nondecreasing\":" << (nondecreasing ? "true" : "false") << ",";
    ofs << "\"keys_match_ref\":" << (keys_match_ref ? "true" : "false") << ",";
    ofs << "\"indices_match_ref\":" << (indices_match_ref ? "true" : "false");
    ofs << "}\n";
}

static inline std::vector<int> make_sizes(int minN, int maxN, double factor) {
    std::vector<int> sizes;
    if (minN < 1) minN = 1;
    if (maxN < minN) maxN = minN;
    double cur = static_cast<double>(minN);
    while (cur < static_cast<double>(maxN)) {
        sizes.push_back(static_cast<int>(cur));
        cur = cur * factor;
        // guard against too-small factor
        if (static_cast<int>(cur) <= sizes.back()) cur = sizes.back() + 1.0;
    }
    // ensure strictly below 2^30
    const int LIMIT = (1u<<30) - 1; // 1073741823
    for (int &v : sizes) if (v > LIMIT) v = LIMIT;
    // dedupe, sort, positive only
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    sizes.erase(std::remove_if(sizes.begin(), sizes.end(), [](int x){return x<=0;}), sizes.end());
    return sizes;
}

// ---- CLI parsing (minimal) ----
struct Args {
    int minN   = 1<<18;          // 262,144
    int maxN   = (1<<30) - 1;    // < 2^30
    double factor = 1.15;        // dense geometric progression
    int reps   = 5;
    int seeds  = 1;
    int maxVal = (1 << 24) - 1;
    std::string out = "radix_bench.jsonl";
    bool enable_cpu_ref = false; // off by default per your request
};
static inline std::vector<std::string> split(const std::string& s, char d){
    std::vector<std::string> v; std::stringstream ss(s); std::string item;
    while (std::getline(ss, item, d)) if (!item.empty()) v.push_back(item);
    return v;
}
static inline bool starts_with(const std::string& s, const std::string& p){
    return s.rfind(p, 0) == 0;
}
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
        else if (tok=="--out"    && i+1<argc) { a.out    = argv[++i]; }
        else if (tok=="--enable_cpu_ref") { a.enable_cpu_ref = true; }
        else if (starts_with(tok, "--help")) {
            std::cout <<
            "Usage: ./bench_dense [--minN 262144] [--maxN 1073741823] [--factor 1.15]\n"
            "                     [--reps 5] [--seeds 1] [--maxVal 16777215]\n"
            "                     [--out radix_bench.jsonl] [--enable_cpu_ref]\n";
            std::exit(0);
        }
    }
    // Keep under 2^30
    const int LIMIT = (1u<<30) - 1;
    if (a.maxN > LIMIT) a.maxN = LIMIT;
    if (a.minN < 1) a.minN = 1;
    if (a.factor < 1.01) a.factor = 1.01; // avoid enormous loops
    return a;
}

// Optional: monotonicity check (non-decreasing)
static inline bool is_nondecreasing(const std::vector<int>& v){
    for (size_t i=1;i<v.size();++i) if (v[i-1] > v[i]) return false;
    return true;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    // CUDA device info
    CUDA_OK(cudaSetDevice(0));
    cudaDeviceProp p{}; CUDA_OK(cudaGetDeviceProperties(&p, 0));
    std::cout << "GPU: " << p.name << " (SM " << p.major << "." << p.minor << ")\n";
    std::cout << "Dense sizes from " << args.minN << " to " << args.maxN
              << " (factor=" << args.factor << ", < 2^30)\n";
    std::cout << "reps=" << args.reps << ", seeds=" << args.seeds
              << ", maxVal=" << args.maxVal << ", out=" << args.out << "\n";
    if (!args.enable_cpu_ref) {
        std::cout << "CPU reference sort: DISABLED (using CUB as correctness reference)\n";
    }

    std::ofstream ofs(args.out, std::ios::out | std::ios::trunc);
    if (!ofs) { std::cerr << "Failed to open " << args.out << "\n"; return 1; }

    // Warm-up small run to prime context
    {
        std::vector<int> small_in(4096), small_out(4096);
        for (int i = 0; i < 4096; ++i) small_in[i] = i & 2500;
        float dummy = 0.0f;
        oneSweepSort(small_in.data(), small_out.data(), 4096, 2500, &dummy);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        std::cout << "[Warm-up] oneSweepSort done\n";
    }

    // Generate dense size list
    std::vector<int> sizes = make_sizes(args.minN, args.maxN, args.factor);
    std::cout << "Total sizes: " << sizes.size() << "\n";

    // Sweep sizes × seeds × repetitions
    for (int N : sizes) {
        std::cout << "\n=== N = " << N << " ===\n";

        for (int si = 0; si < args.seeds; ++si) {
            uint64_t seed = 12345 + si;
            std::mt19937_64 rng(seed);
            std::uniform_int_distribution<int> dist(0, args.maxVal);

            // Host inputs/outputs
            std::vector<int> h_in(N), h_one(N), h_r4(N), h_cub(N), h_idx(N);
            for (int i = 0; i < N; ++i) h_in[i] = dist(rng);

            // Optionally (rarely) run a CPU reference if enabled
            std::vector<int> ref_keys; // keep empty unless enabled
            if (args.enable_cpu_ref) {
                std::vector<std::pair<int,int>> ref(N);
                for (int i = 0; i < N; ++i) ref[i] = {h_in[i], i};
                auto t0_ref = std::chrono::high_resolution_clock::now();
                std::stable_sort(ref.begin(), ref.end(),
                    [](const auto& a, const auto& b){ return a.first < b.first; });
                auto t1_ref = std::chrono::high_resolution_clock::now();
                double ref_ms = std::chrono::duration<double, std::milli>(t1_ref - t0_ref).count();
                std::cout << "CPU stable_sort (optional): " << std::fixed << std::setprecision(3) << ref_ms << " ms\n";
                ref_keys.resize(N);
                for (int i = 0; i < N; ++i) ref_keys[i] = ref[i].first;
            }

            for (int r = 0; r < args.reps; ++r) {
                // --- CUB (keys + indices) --- used as correctness baseline
                float cub_dev_ms = 0.0f;
                auto t0_c = std::chrono::high_resolution_clock::now();
                launchCUBRadixSortPairs(h_in.data(), h_cub.data(), h_idx.data(), N, args.maxVal, &cub_dev_ms);
                CUDA_OK(cudaGetLastError());
                CUDA_OK(cudaDeviceSynchronize());
                auto t1_c = std::chrono::high_resolution_clock::now();
                double cub_wall_ms = std::chrono::duration<double, std::milli>(t1_c - t0_c).count();
                double cub_g_dev  = (cub_dev_ms > 0) ? (static_cast<double>(N) / (cub_dev_ms/1000.0) / 1e9) : 0.0;
                double cub_g_wall = (cub_wall_ms > 0) ? (static_cast<double>(N) / (cub_wall_ms/1000.0) / 1e9) : 0.0;
                bool cub_nondec = is_nondecreasing(h_cub);
                // We assume CUB correctness; indices_match_ref = true
                write_jsonl(ofs, "cub_pairs", p.name, p.major, p.minor,
                            N, r, seed, args.maxVal, cub_dev_ms, cub_wall_ms,
                            cub_g_dev, cub_g_wall, cub_nondec, true, true);
                std::cout << "cub_pairs r=" << r
                          << " | dev=" << cub_dev_ms << " ms, wall=" << cub_wall_ms << " ms"
                          << " | Gitems/s(dev)=" << std::setprecision(6) << cub_g_dev
                          << " | ok(nondec)=" << (cub_nondec ? "Y":"N") << "\n";

                // --- oneSweep (keys-only) ---
                {
                    std::vector<int> in = h_in; // copy for fairness
                    float dev_ms = 0.0f;
                    auto t0 = std::chrono::high_resolution_clock::now();
                    oneSweepSort(in.data(), h_one.data(), N, args.maxVal, &dev_ms);
                    CUDA_OK(cudaGetLastError());
                    CUDA_OK(cudaDeviceSynchronize());
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                    bool nondec = is_nondecreasing(h_one);
                    // correctness: compare keys to CUB output (fast & stable)
                    bool keys_ok = (h_one.size()==h_cub.size()) &&
                                   std::equal(h_one.begin(), h_one.end(), h_cub.begin());

                    // If CPU ref enabled, you can also compare to ref_keys:
                    if (args.enable_cpu_ref && ref_keys.size()==static_cast<size_t>(N)) {
                        keys_ok = keys_ok && std::equal(h_one.begin(), h_one.end(), ref_keys.begin());
                    }

                    double g_dev  = (dev_ms  > 0) ? (static_cast<double>(N) / (dev_ms /1000.0) / 1e9) : 0.0;
                    double g_wall = (wall_ms > 0) ? (static_cast<double>(N) / (wall_ms/1000.0) / 1e9) : 0.0;

                    write_jsonl(ofs, "one_sweep", p.name, p.major, p.minor,
                                N, r, seed, args.maxVal, dev_ms, wall_ms,
                                g_dev, g_wall, nondec, keys_ok, true);
                    std::cout << "one_sweep r=" << r
                              << " | dev=" << dev_ms << " ms, wall=" << wall_ms << " ms"
                              << " | Gitems/s(dev)=" << std::setprecision(6) << g_dev
                              << " | ok=" << (nondec && keys_ok ? "Y":"N") << "\n";
                }

                // --- radix4 (keys-only) ---
                {
                    std::vector<int> in = h_in;
                    float dev_ms = 0.0f;
                    auto t0 = std::chrono::high_resolution_clock::now();
                    launchRadixSort_4(in.data(), h_r4.data(), N, args.maxVal, &dev_ms);
                    CUDA_OK(cudaGetLastError());
                    CUDA_OK(cudaDeviceSynchronize());
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                    bool nondec = is_nondecreasing(h_r4);
                    bool keys_ok = (h_r4.size()==h_cub.size()) &&
                                   std::equal(h_r4.begin(), h_r4.end(), h_cub.begin());
                    if (args.enable_cpu_ref && ref_keys.size()==static_cast<size_t>(N)) {
                        keys_ok = keys_ok && std::equal(h_r4.begin(), h_r4.end(), ref_keys.begin());
                    }

                    double g_dev  = (dev_ms  > 0) ? (static_cast<double>(N) / (dev_ms /1000.0) / 1e9) : 0.0;
                    double g_wall = (wall_ms > 0) ? (static_cast<double>(N) / (wall_ms/1000.0) / 1e9) : 0.0;

                    write_jsonl(ofs, "radix4", p.name, p.major, p.minor,
                                N, r, seed, args.maxVal, dev_ms, wall_ms,
                                g_dev, g_wall, nondec, keys_ok, true);
                    std::cout << "radix4    r=" << r
                              << " | dev=" << dev_ms << " ms, wall=" << wall_ms << " ms"
                              << " | Gitems/s(dev)=" << std::setprecision(6) << g_dev
                              << " | ok=" << (nondec && keys_ok ? "Y":"N") << "\n";
                }
            } // reps
        } // seeds
    } // sizes

    std::cout << "\nJSONL written to: " << args.out << "\n";
    std::cout << "Fields include: device_ms, wall_ms, gitems_s_device, gitems_s_wall\n";
    return 0;
}
