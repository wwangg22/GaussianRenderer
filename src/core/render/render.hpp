#pragma once
#include "gaussians.hpp"
#include "camera.hpp"

struct TilingInformation {
    int num_tile_y;
    int num_tile_x;
    size_t H;
    size_t W;
    size_t* tile_id_counts;

    TilingInformation(int ny, int nx, size_t h, size_t w) : num_tile_y(ny), num_tile_x(nx), H(h), W(w) {
        tile_id_counts = new size_t[num_tile_y * num_tile_x];
        std::memset(tile_id_counts, 0, sizeof(size_t) * num_tile_y * num_tile_x);
    }
    ~TilingInformation() {
        delete[] tile_id_counts;
    }
};

std::vector<Gaussian> frustum_cull(std::vector<Gaussian>& g, Camera& cam, const float& treshold);
void transformAndTileGaussians(std::vector<Gaussian>& g, std::vector<lightWeightGaussian>& out, Camera& cam,
                                TilingInformation& tile_info, float k);