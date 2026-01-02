#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <cstdlib>

enum class SlotType { X, Y, Z, Normal, SH_DC, SH_REST, Opacity, Scale, Rot, Skip };

struct Property {
    SlotType type;
    int index; // for properties that have multiple indices, like normals, SH, scale, rot
};
struct Gaussian {
    float x, y, z;
    float normals[3];
    float sh[27]; //we can expand later
    float color[3];
    float opacity;
    float scale[3];
    float rot[4];
    int aabb[4]; // x_min, y_min, x_max, y_max
    int px_x, px_y;
    uint64_t radix_id;
    float X, Y, Z;

    float inv_covar[4]; // 2x2 inverse covariance matrix stored in row-major order
};

struct lightWeightGaussian {
    uint64_t radix_id; // upper 32 bits tile_id, lower 32 bits depth_id
    uint32_t gaussian_id; // index in the original Gaussian array
};


struct TilingInformation {
    int num_tile_y;
    int num_tile_x;
    int H;
    int W;
    std::vector<int> tile_id_offset;
    int height_stride;
    int width_stride;

    TilingInformation(int ny, int nx, int h, int w) : num_tile_y(ny), num_tile_x(nx), H(h), W(w), tile_id_offset(ny*nx) {
        width_stride  = std::max<int>(1, (W + num_tile_x - 1) / num_tile_x);
        height_stride = std::max<int>(1, (H + num_tile_y - 1) / num_tile_y);
    }
    void resize(int h, int w, int num_tile_x_, int num_tile_y_) {
        this->H = h;
        this->W = w;
        this->num_tile_x = num_tile_x_;
        this->num_tile_y = num_tile_y_;
        width_stride  = std::max<int>(1, (W + num_tile_x - 1) / num_tile_x);
        height_stride = std::max<int>(1, (H + num_tile_y - 1) / num_tile_y);
    };
    ~TilingInformation() = default;
};

void storeGaussianFromProperty(const Property& prop, Gaussian& g, float value);
std::vector<Gaussian> loadGaussiansFromPly(const std::string& filename);