#pragma once
#define RADIX_BITS 8
#define RADIX 256

#define TILE_SIZE 2048
#include "gaussians.hpp"
extern "C" void oneSweep3DGaussianSort(lightWeightGaussian* d_in, 
                                       int N, 
                                       int num_bits,
                                       float* kernel_ms);