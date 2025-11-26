#pragma once
#define RADIX_BITS 8
#define RADIX 256

#define TILE_SIZE 2048
#include "gaussians.hpp"
#include "camera.hpp"
extern "C" void oneSweep3DGaussianSort(lightWeightGaussian* d_in, 
                                       int N, 
                                       int num_bits,
                                       float* kernel_ms);

extern "C" void renderGaussiansCUDA(float* d_out_pixels, 
                                 TilingInformation* d_tile_info, 
                                 Gaussian* d_gaussians, 
                                 lightWeightGaussian* d_sorted_gaussians,
                                 int num_gaussians,
                                 int num_lwg,
                                 float* kernel_ms);
extern "C" void renderGaussiansNoTilingCUDA(float* out_pixels, 
                                 TilingInformation* tile_info, 
                                 Gaussian* gaussians, 
                                 lightWeightGaussian* sorted_gaussians,
                                 int num_gaussians,
                                 float* kernel_ms);

extern "C" void preprocessCUDAGaussians(Gaussian* d_gaussians,
    int num_gaussians,
    Camera cam,
    TilingInformation* d_tile_info,
    float k);