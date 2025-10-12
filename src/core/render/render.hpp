#include "gaussians.hpp"

struct TilingInformation {
    size_t num_tile_y;
    size_t num_tile_x;
    size_t H;
    size_t W;
    size_t* tile_id_counts;

    TilingInformation(size_t ny, size_t nx, size_t h, size_t w) : num_tile_y(ny), num_tile_x(nx), H(h), W(w) {
        tile_id_counts = new size_t[num_tile_y * num_tile_x];
        std::memset(tile_id_counts, 0, sizeof(size_t) * num_tile_y * num_tile_x);
    }
    ~TilingInformation() {
        delete[] tile_id_counts;
    }
};

struct lightWeightGaussian {
    uint64_t radix_id; // upper 32 bits tile_id, lower 32 bits depth_id
    uint32_t gaussian_id; // index in the original Gaussian array
};

void frustum_cull(std::vector<Gaussian>& g,std::vector<Gaussian>& out, float* planes, float& treshold);
void transformAndTileGaussians(std::vector<Gaussian>& g, std::vector<uint64_t>& radix_ids, float* M, size_t num_tile_y, size_t num_tile_x, size_t H, size_t W);