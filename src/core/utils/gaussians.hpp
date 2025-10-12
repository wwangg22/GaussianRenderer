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
    float aabb[4]; // x_min, y_min, x_max, y_max
    uint64_t radix_id;

    float covar[4]; // 2x2 covariance matrix stored in row-major order


    Gaussian();
};


std::vector<Gaussian> loadGaussiansFromPly(const std::string& filename);