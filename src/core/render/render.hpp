#pragma once
#include "gaussians.hpp"
#include "camera.hpp"

struct Vertex {
    float pos[4];     // in_vertex
    float uv[4];      // in_texcoord (z,w unused)
    float col[4];     // in_color
    
};
struct OrbitControls {
    bool   dragging = false;
    double lastX    = 0.0;
    double lastY    = 0.0;
    float zoomSpeed   = 0.1f; 

    // tweak these until it feels right
    float orbitSpeedX = 0.3f;  // degrees per pixel horizontally (azimuth)
    float orbitSpeedY = 0.3f;  // degrees per pixel vertically (elevation)
};

std::vector<Gaussian> frustum_cull(std::vector<Gaussian>& g, Camera& cam, const float& treshold);
void transformAndTileGaussians(std::vector<Gaussian>& g, std::vector<lightWeightGaussian>& out, Camera& cam,
                                TilingInformation& tile_info, float k);
