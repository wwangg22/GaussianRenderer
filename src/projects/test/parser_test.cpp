#include "gaussians.hpp"
 
int main(int argc, char* argv[]){
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_ply_file>\n";
        return 1;
    }

    std::string filename = argv[1];
    std::vector<Gaussian> gaussians = loadGaussiansFromPly(filename);
}