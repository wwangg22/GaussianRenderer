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

    Gaussian();
};


std::vector<Gaussian> loadGaussiansFromPly(const std::string& filename);