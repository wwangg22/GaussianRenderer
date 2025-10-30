#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <cstdlib>

struct Gaussian {
    float x, y, z;
    float normals[3];
    float sh[27]; //we can expand later
    float opacity;
    float scale[3];
    float rot[4];
    int aabb[4]; // x_min, y_min, x_max, y_max
    uint64_t radix_id;

    float covar[4]; // 2x2 covariance matrix stored in row-major order


    Gaussian();
};

struct lightWeightGaussian {
    uint64_t radix_id; // upper 32 bits tile_id, lower 32 bits depth_id
    uint32_t gaussian_id; // index in the original Gaussian array
};

std::vector<Gaussian> loadGaussiansFromPly(const std::string& filename);