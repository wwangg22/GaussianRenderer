#pragma once
#include "gaussians.hpp"
#include "camera.hpp"

std::vector<Gaussian> frustum_cull(std::vector<Gaussian>& g, Camera& cam, const float& treshold);
void transformAndTileGaussians(std::vector<Gaussian>& g, std::vector<lightWeightGaussian>& out, Camera& cam,
                                TilingInformation& tile_info, float k);
void drawScreen(float* pixel_out);