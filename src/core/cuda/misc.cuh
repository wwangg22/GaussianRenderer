#pragma once
#include <cuda_runtime.h>
#include "gaussians.hpp" 

Gaussian *loadGaussianCudaFromPly(const std::string& filename, int* out_numGaussians);