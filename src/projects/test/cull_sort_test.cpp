#include <chrono>
#include <cuda_runtime.h>

#include "render.cuh"
#include "misc.cuh"
#include "gaussians.hpp"
#include "render.hpp"
#include "camera.hpp"
#include "canvas.hpp"


const int width = 2000;
const int height = 1500;
int main(int argc, char* argv[]){
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_ply_file>\n";
        return 1;
    }
    float p[3] = {-1.5f, -1.5f, -3.0f};
    Camera cam;
    cam.w_up[0] = 0.0f; cam.w_up[1] = -1.0f; cam.w_up[2] = 0.0f;
    cam.setFovY(90.0f);
    cam.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
    cam.setClippingPlanes(0.1f, 100.0f);
    cam.setPosition(p);
    cam.updateCameraMatrices();
    cam.updateFrustumPlanes();
    
    std::string filename = argv[1];
    int numGaussians = 0;
    // std::vector<Gaussian> gaussians = loadGaussiansFromPly(filename);
    Gaussian * d_gaussians = loadGaussianCudaFromPly(filename, &numGaussians); //this is cuda pointer
    if (d_gaussians == nullptr) {
        std::cerr << "Failed to load gaussians from PLY file: " << filename << std::endl;
        return 1;
    }
    std::cout << "Loaded " << numGaussians << " gaussians from " << filename << std::endl;
    
    Canvas canvas(height, width, 64, 64);
    canvas.init();
    canvas.cam = &cam;

    while (true) {
        preprocessCUDAGaussians(d_gaussians, canvas.d_out_pixels.data(), numGaussians, cam, 
                canvas.tile_info.num_tile_y, canvas.tile_info.num_tile_x, canvas.tile_info.width_stride,
                canvas.tile_info.height_stride, canvas.tile_info.W, canvas.tile_info.H, 3.0f);
        canvas.draw(canvas.d_out_pixels.data());
    }
 
    cudaFree(d_gaussians);


    return 0;
}