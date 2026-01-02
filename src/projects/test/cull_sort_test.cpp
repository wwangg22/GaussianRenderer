#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

#include "render.cuh"
#include "misc.cuh"
#include "gaussians.hpp"
#include "render.hpp"
#include "camera.hpp"
#include "canvas.hpp"


const int width = 2000;
const int height = 1500;


double ema_ms = 0.0;
const double alpha = 0.1; // smoothing

int main(int argc, char* argv[]){
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_ply_file>\n";
        return 1;
    }
    float p[3] = {-1.5f, -1.5f, -3.0f};
    Camera cam;
    cam.w_up[0] = 0.0f; cam.w_up[1] = -1.0f; cam.w_up[2] = 0.0f;
    cam.setFovY(120.0f);
    cam.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
    cam.setClippingPlanes(2.5f, 100.0f);
    cam.setPosition(p);
    cam.updateCameraMatrices();
    cam.updateFrustumPlanes();
    
    std::string filename = argv[1];
    int numGaussians = 0;
    // std::vector<Gaussian> gaussians = loadGaussiansFromPly(filename);
    // Gaussian * d_gaussians = loadGaussianCudaFromPly(filename, &numGaussians); //this is cuda pointer
    // if (d_gaussians == nullptr) {
    //     std::cerr << "Failed to load gaussians from PLY file: " << filename << std::endl;
    //     return 1;
    // }
    // std::cout << "Loaded " << numGaussians << " gaussians from " << filename << std::endl;
    int num_tile_x = 50;
    int num_tile_y = 50;
    Canvas canvas(height, width, num_tile_x, num_tile_y);
    canvas.cam = &cam;
    canvas.init();
    canvas.loadGaussians(filename);
    

    while (true) {
        auto t0 = std::chrono::high_resolution_clock::now();
        canvas.render();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        ema_ms = (ema_ms == 0.0) ? ms : (alpha * ms + (1.0 - alpha) * ema_ms);

        static int frame = 0;
        if ((frame++ % 60) == 0) {
            printf("frame: %.3f ms  (%.1f FPS)\n", ema_ms, 1000.0 / ema_ms);
        }
    }
 

    return 0;
}