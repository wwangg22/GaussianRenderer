#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <cstring>
#include <cstdlib>
#include "misc.cuh"
#include "gaussians.hpp"

Gaussian *loadGaussianCudaFromPly(const std::string& filename, int* out_numGaussians){
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {};
    }

    // read total number to elements in file
    // we will assume only vertices
    std::string format;
    std::string line;
    std::string prefix = "format ";
    while (std::getline(file, line)) {
        if (line.substr(0, prefix.size()) == prefix) {
            format = line.substr(prefix.size());
            break;
        }
    }
    prefix = "element vertex ";
    while (std::getline(file, line)) {
        if (line.substr(0, prefix.size()) == prefix) {
            break;
        }
    }
    int numGaussians = std::stoi(line.substr(prefix.size()));
    *out_numGaussians = numGaussians;

    prefix = "property ";
    std::vector<Property> properties;
    // right now we only support 27 SH coefficients
    std::unordered_set<std::string> neededProps = {
        "x", "y", "z",
        "nxx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        // First 24 f_rest
        "f_rest_0", "f_rest_1", "f_rest_2", "f_rest_3",
        "f_rest_4", "f_rest_5", "f_rest_6", "f_rest_7",
        "f_rest_8", "f_rest_9", "f_rest_10", "f_rest_11",
        "f_rest_12", "f_rest_13", "f_rest_14", "f_rest_15",
        "f_rest_16", "f_rest_17", "f_rest_18", "f_rest_19",
        "f_rest_20", "f_rest_21", "f_rest_22", "f_rest_23",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3"
    };
    while (std::getline(file, line)){
        if (line == "end_header") break;

        if (line.substr(0, prefix.size()) == prefix) {
            std::istringstream iss(line.substr(prefix.size()));
            std::string type, name;
            iss >> type >> name; 
            if (name == "x") properties.push_back({SlotType::X, 0});
            else if (name == "y") properties.push_back({SlotType::Y, 0});
            else if (name == "z") properties.push_back({SlotType::Z, 0});
            else if (name == "nxx") properties.push_back({SlotType::Normal, 0});
            else if (name == "ny")  properties.push_back({SlotType::Normal, 1});
            else if (name == "nz")  properties.push_back({SlotType::Normal, 2});
            else if (name == "f_dc_0") properties.push_back({SlotType::SH_DC, 0});
            else if (name == "f_dc_1") properties.push_back({SlotType::SH_DC, 1});
            else if (name == "f_dc_2") properties.push_back({SlotType::SH_DC, 2});
            else if (name.rfind("f_rest_",0) == 0) {
                int idx = std::stoi(name.substr(7));
                if (idx < 24) properties.push_back({SlotType::SH_REST, idx});
                else properties.push_back({SlotType::Skip, 0});
            }
            else if (name == "opacity") properties.push_back({SlotType::Opacity, 0});
            else if (name.rfind("scale_",0) == 0) {
                int idx = std::stoi(name.substr(6));
                properties.push_back({SlotType::Scale, idx});
            }
            else if (name.rfind("rot_",0) == 0) {
                int idx = std::stoi(name.substr(4));
                properties.push_back({SlotType::Rot, idx});
            }
            else properties.push_back({SlotType::Skip, 0});
        }
    }

    // start reading data
    if (format == "binary_little_endian 1.0") {
        std::vector<Gaussian> h_gaussians(numGaussians);

        for (int i = 0; i < numGaussians; ++i) {
            Gaussian g{};
            for (const auto& prop : properties) {
                float value;
                file.read(reinterpret_cast<char*>(&value), sizeof(float));
                storeGaussianFromProperty(prop, g, value);
            }

            h_gaussians[i] = g;
        }

        Gaussian* cuda_gaussians = nullptr;
        cudaError_t err = cudaMalloc(&cuda_gaussians,
                                    sizeof(Gaussian) * numGaussians);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memory allocation failed: "
                    << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }

        err = cudaMemcpy(cuda_gaussians,
                        h_gaussians.data(),
                        sizeof(Gaussian) * numGaussians,
                        cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memcpy failed: "
                    << cudaGetErrorString(err) << std::endl;
            cudaFree(cuda_gaussians);
            return nullptr;
        }

        return cuda_gaussians;
    } else if (format != "ascii 1.0") {
        std::cerr << "Unsupported PLY format: " << format << std::endl;
        return nullptr;
    } else {
        std::cerr << "Unsupported PLY format: " << format << std::endl;
        return nullptr;
    }
}