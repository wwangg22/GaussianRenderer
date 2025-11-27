#include <chrono>
#include <cuda_runtime.h>

#include "render.cuh"
#include "misc.cuh"
#include "gaussians.hpp"
#include "render.hpp"
#include "camera.hpp"
#include "canvas.hpp"


const int width = 1000;
const int height = 750;
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
    
    TilingInformation tile_info(16, 16, height, width);
    std::vector<float> d_out_pixels(height * width * 3);
    Canvas canvas(height, width);
    canvas.init();
    canvas.cam = &cam;

    while (true) {
        preprocessCUDAGaussians(d_gaussians, d_out_pixels.data(), numGaussians, cam, &tile_info, 3.0f);
        canvas.draw(d_out_pixels.data());
    }
    
   
    
    // TilingInformation tile_info(16, 16, 600, 800);

    // while (true) {
    //     std::vector<Gaussian> culled_gaussians = frustum_cull(gaussians, cam, 4.0f);
    //     std::vector<lightWeightGaussian> tiled_gaussians;
    //     transformAndTileGaussians(culled_gaussians, tiled_gaussians, cam, tile_info, 4.0f);
    //     std::cout << "tile gaussians size: " << tiled_gaussians.size() << std::endl;
    //     std::cout << "Tiled " << tiled_gaussians.size() << " gaussians." << std::endl;
    //     float kernel_ms;
    //     // print first few tiled gaussians
    //     // for (int i = 0; i < 10 && i < tiled_gaussians.size(); i++) {
    //     //     std::cout << "tiled_gaussians[" << i << "]: radix_id = " << tiled_gaussians[i].radix_id 
    //     //               << ", gaussian_id = " << tiled_gaussians[i].gaussian_id << std::endl;
    //     // }
    //     oneSweep3DGaussianSort(tiled_gaussians.data(), 
    //                         tiled_gaussians.size(), 
    //                         48, // using 64 bits for radix sort
    //                         &kernel_ms);
    //     // for (int i = 0; i < 10 && i < tiled_gaussians.size(); i++) {
    //     //     std::cout << "tiled_gaussians[" << i << "]: radix_id = " << tiled_gaussians[i].radix_id 
    //     //               << ", gaussian_id = " << tiled_gaussians[i].gaussian_id << std::endl;
    //     // }
    //     std::cout << "Kernel execution time: " << kernel_ms << " ms." << std::endl;



    //     // renderGaussiansCUDA(d_out_pixels, &tile_info, culled_gaussians.data(), tiled_gaussians.data(), culled_gaussians.size(), tiled_gaussians.size(), &kernel_ms);
    //     renderGaussiansNoTilingCUDA(d_out_pixels.data(), &tile_info, culled_gaussians.data(), tiled_gaussians.data(), tiled_gaussians.size(), &kernel_ms);
        
    //     // std::cout << "Render kernel execution time: " << kernel_ms << " ms." << std::endl;
    //     // for (int i = 0; i < 10; i++) {
    //     //     std::cout << d_out_pixels[0 * height*width + i] << " " << d_out_pixels[1 * height*width + i] << " " << d_out_pixels[2 * height*width + i] << "\n";
    //     // }
    //     // print render time
    //     std::cout << "Rendering execution time: " << kernel_ms << " ms." << std::endl;

    //     // print out first 10 pixel values
    //     // std::cout << "First 10 pixel values:\n";
    //     // for (int i = 0; i < 10; i++) {
    //     //     std::cout << d_out_pixels[0 * height*width + i] << " " << d_out_pixels[1 * height*width + i] << " " << d_out_pixels[2 * height*width + i] << "\n";
    //     // }

    //     canvas.draw(d_out_pixels.data());
    // }    
    cudaFree(d_gaussians);


    return 0;
}