// Benchmark host-side driver for the CUDA matrix multiply

#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

extern "C" void launchMatrixMul(float* d_A, float* d_B, float* d_result, int N, int threadsPerBlockX, int threadsPerBlockY);

static const char* cudaErrorToString(cudaError_t err) {
    return cudaGetErrorString(err);
}

int main(int argc, char** argv) {
    // Select device 0 by default
    int dev = 0;
    cudaError_t err = cudaSetDevice(dev);
    if (err != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: " << cudaErrorToString(err) << "\n";
        return 1;
    }

    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, dev);
    std::cout << "GPU: " << p.name << " (SM " << p.major << "." << p.minor << ")\n";

    // Parse N and fill mode
    int N = 8192; // default size; override via argv[1]
    int threadsPerBlockX=16;
    int threadsPerBlockY=16;
    std::string mode = "ones"; // fill mode: ones | random
    if (argc >= 2) {
        N = std::atoi(argv[1]);
        if (N <= 0) {
            std::cerr << "Invalid N; please pass a positive integer." << std::endl;
            return 1;
        }
    }
    if (argc >= 4) {
        threadsPerBlockX = std::atoi(argv[2]);
        threadsPerBlockY = std::atoi(argv[3]);
        if (threadsPerBlockX <= 0 || threadsPerBlockY <= 0) {
            std::cerr << "Invalid block dimensions; please pass positive integers." << std::endl;
            return 1;
        } 
    }

    size_t elements = static_cast<size_t>(N) * static_cast<size_t>(N);
    size_t bytes = elements * sizeof(float);
    double total_bytes = static_cast<double>(bytes) * 3.0; // A, B, C on device

    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Matrix size: N=" << N << " (" << (elements / 1e6) << "M elems)\n";
    std::cout << "Approx GPU alloc (A,B,C): " << (total_bytes / (1024.0 * 1024.0)) << " MiB\n";
    std::cout << "GPU mem: free " << (free_b / (1024.0 * 1024.0)) << " MiB / total "
              << (total_b / (1024.0 * 1024.0)) << " MiB\n";
    std::cout << "Fill mode: " << mode << "\n";
    std::cout << "Threads per block: " << threadsPerBlockX << " x " << threadsPerBlockY << "\n";

    // Allocate host buffers
    std::vector<float> A(elements);
    std::vector<float> B(elements);
    std::vector<float> C(elements, 0.0f);

    if (mode == "ones") {
        std::fill(A.begin(), A.end(), 1.0f);
        std::fill(B.begin(), B.end(), 1.0f);
    } else {
        std::mt19937 rng(123);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < elements; ++i) {
            A[i] = dist(rng);
            B[i] = dist(rng);
        }
    }

    // Warm-up (small) to initialize CUDA context without impacting timing
    {
        std::vector<float> wA(16 * 16, 1.0f), wB(16 * 16, 1.0f), wC(16 * 16, 0.0f);
        launchMatrixMul(wA.data(), wB.data(), wC.data(), 16, 16, 16);
    }

    std::cout << "Running multiply..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    launchMatrixMul(A.data(), B.data(), C.data(), N, threadsPerBlockX, threadsPerBlockY);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Check for any CUDA error surfaced
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after launchMatrixMul: " << cudaErrorToString(err) << "\n";
    }

    std::chrono::duration<double, std::milli> ms = t1 - t0;
    double sec = ms.count() / 1000.0;
    // Naive GEMM ~ 2*N^3 FLOPs
    long double flops = 2.0L * static_cast<long double>(N) * N * N;
    double gflops = static_cast<double>(flops / 1.0e9L) / sec;

    std::cout << std::setprecision(3);
    std::cout << "Time: " << ms.count() << " ms\n";
    std::cout << "Throughput: " << gflops << " GFLOP/s\n";

    // Lightweight correctness spot-check
    bool ok = true;
    if (mode == "ones") {
        const float expected = static_cast<float>(N);
        for (int i = 0; i < 5; ++i) {
            size_t idx = (static_cast<size_t>(i) * (elements / 5)) % elements;
            float v = C[idx];
            if (std::abs(v - expected) > 1e-2f) {
                ok = false;
                break;
            }
        }
    }

    std::cout << "Check: " << (ok ? "pass" : "FAIL") << "\n";
    return ok ? 0 : 2;
}
