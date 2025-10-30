#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <unordered_set>
#include "gaussians.hpp"

enum class SlotType { X, Y, Z, Normal, SH_DC, SH_REST, Opacity, Scale, Rot, Skip };

struct Property {
    SlotType type;
    int index; // for properties that have multiple indices, like normals, SH, scale, rot
};
Gaussian::Gaussian() : x(0), y(0), z(0), opacity(0), radix_id(0){
        std::fill(std::begin(normals), std::end(normals), 0.0f);
        std::fill(std::begin(aabb), std::end(aabb), 0.0f);
        std::fill(std::begin(sh), std::end(sh), 0.0f);
        std::fill(std::begin(scale), std::end(scale), 0.0f);
        std::fill(std::begin(rot), std::end(rot), 0.0f);
 }

void storeGaussianFromProperty(const Property& prop, Gaussian& g, float value){
    switch (prop.type) {
        case SlotType::X: g.x = value; break;
        case SlotType::Y: g.y = value; break;
        case SlotType::Z: g.z = value; break;
        case SlotType::Normal: g.normals[prop.index] = value; break;
        case SlotType::SH_DC: g.sh[prop.index] = value; break;
        case SlotType::SH_REST: g.sh[3 + prop.index] = value; break;
        case SlotType::Opacity: g.opacity = value; break;
        case SlotType::Scale: g.scale[prop.index] = value; break;
        case SlotType::Rot: g.rot[prop.index] = value; break;
        default: break; // Skip
    }
}

std::vector<Gaussian> loadGaussiansFromPly(const std::string& filename) {
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
    std::cout << "Number of gaussians: " << numGaussians;

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
    std::vector<Gaussian> gaussians;

    if (format == "binary_little_endian 1.0") {
        for (int i = 0; i < numGaussians; ++i) {
            Gaussian g;
            for (const auto& prop : properties) {
                float value;
                file.read(reinterpret_cast<char*>(&value), sizeof(float));
                storeGaussianFromProperty(prop, g, value);
            }
            // Only add if opacity > 0
            // if (g.opacity > 0.0f) {
            //     gaussians.push_back(g);
            // }
            gaussians.push_back(g);
            if (i < 10){
                // print gaussian position
                std::cout << "\nGaussian " << i << ": ";
                std::cout << g.x << ", " << g.y << ", " << g.z;
            }
        }
        // std::cout << "example gaussian: "<< gaussians[1].x << ", " << gaussians[1].y << ", " << gaussians[1].z << ", " << gaussians[1].opacity << std::endl;
        return gaussians;
    } else if (format != "ascii 1.0") {
        std::cerr << "Unsupported PLY format: " << format << std::endl;
        return gaussians;
    } else {
        std::cerr << "Unsupported PLY format: " << format << std::endl;
        return gaussians;
    }
}