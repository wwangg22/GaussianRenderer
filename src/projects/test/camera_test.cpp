#include <iostream>
#include "camera.hpp"

int main(){
    Camera cam;

    cam.setFovY(45.0f);
    std::cout << "testing camera! " << std::endl;
    return 0;
}