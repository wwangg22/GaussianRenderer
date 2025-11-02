#include <chrono>
#include <cuda_runtime.h>

#include "render.cuh"
#include "gaussians.hpp"
#include "render.hpp"
#include "camera.hpp"


int main(int argc, char* argv[]){
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_ply_file>\n";
        return 1;
    }
    float p[3] = {-1.5f, -1.5f, -3.0f};
    Camera cam;
    cam.w_up[0] = 0.0f; cam.w_up[1] = -1.0f; cam.w_up[2] = 0.0f;
    cam.setFovY(90.0f);
    cam.setAspectRatio(800.0f / 600.0f);
    cam.setClippingPlanes(0.1f, 100.0f);
    cam.setPosition(p);
    cam.updateCameraMatrices();
    cam.updateFrustumPlanes();

    std::string filename = argv[1];
    std::vector<Gaussian> gaussians = loadGaussiansFromPly(filename);

    std::cout << "Loaded " << gaussians.size() << " gaussians from " << filename << std::endl;

    // start timing
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Gaussian> culled_gaussians = frustum_cull(gaussians, cam, 3.1f);
    // end timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "After culling: " << culled_gaussians.size() << " gaussians remain." << std::endl;
    std::cout << "Culling took " << elapsed.count() << " ms." << std::endl;

    std::vector<lightWeightGaussian> tiled_gaussians;
    TilingInformation tile_info(16, 16, 600, 800);
    transformAndTileGaussians(culled_gaussians, tiled_gaussians, cam, tile_info, 3.0f);
    std::cout << "Tiled " << tiled_gaussians.size() << " gaussians." << std::endl;
    float kernel_ms;
    oneSweep3DGaussianSort(tiled_gaussians.data(), 
                          tiled_gaussians.size(), 
                          64, // using 64 bits for radix sort
                          &kernel_ms);

    std::cout << "Kernel execution time: " << kernel_ms << " ms." << std::endl;

    float *d_out_pixels = (float*) malloc(800 * 600 * 3 * sizeof(float));


    // renderGaussiansCUDA(d_out_pixels, &tile_info, culled_gaussians.data(), tiled_gaussians.data(), culled_gaussians.size(), tiled_gaussians.size(), &kernel_ms);
    renderGaussiansNoTilingCUDA(d_out_pixels, &tile_info, culled_gaussians.data(), tiled_gaussians.data(), tiled_gaussians.size(), &kernel_ms);
    
    std::cout << "Render kernel execution time: " << kernel_ms << " ms." << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << d_out_pixels[0 * 800*600 + i] << " " << d_out_pixels[1 * 800*600 + i] << " " << d_out_pixels[2 * 800*600 + i] << "\n";
    }
    // print render time
    std::cout << "Rendering execution time: " << kernel_ms << " ms." << std::endl;

    // print out first 10 pixel values
    std::cout << "First 10 pixel values:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << d_out_pixels[0 * 800*600 + i] << " " << d_out_pixels[1 * 800*600 + i] << " " << d_out_pixels[2 * 800*600 + i] << "\n";
    }
    drawScreen(d_out_pixels);

    return 0;
}