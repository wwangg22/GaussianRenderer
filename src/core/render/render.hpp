#pragma once
#include <glad/glad.h> 
#include <GLFW/glfw3.h>
#include "gaussians.hpp"
#include "camera.hpp"

struct Vertex {
    float pos[4];     // in_vertex
    float uv[4];      // in_texcoord (z,w unused)
    float col[4];     // in_color
    
};
struct OrbitControls {
    Camera* cam = nullptr;

    bool   dragging = false;
    double lastX    = 0.0;
    double lastY    = 0.0;

    // tweak these until it feels right
    float orbitSpeedX = 0.3f;  // degrees per pixel horizontally (azimuth)
    float orbitSpeedY = 0.3f;  // degrees per pixel vertically (elevation)
};
class Canvas {
public: 
    int width;
    int height;
    GLFWwindow* window;
    Camera* cam;
    OrbitControls controls;
 
    Canvas(int width, int height);
    void init();
    void draw(float* pixel_out);
    void MouseCallback(GLFWwindow* window, int button, int action, int mods);
    void CursorPosCallback(GLFWwindow* window, double xpos, double ypos);

private:
    unsigned int VBO;
    unsigned int VAO;

    Vertex quad[6];
    unsigned int vertexShader;
    unsigned int fragmentShader;
    unsigned int shaderProgram;

    GLuint ssbo=0;

    const char* vertexShaderSource = "#version 450\n"
        "layout(location = 0) in vec4 in_vertex;   /**< Input vertex coordinates */\n"
        "layout(location = 1) in vec4 in_texcoord; /**< Input texture coordinates */\n"
        "layout(location = 2) in vec4 in_color;    /**< Input colour value */\n"
        "\n"
        "out vec4 texcoord;                        /**< Output texture coordinates */\n"
        "out vec4 color;                           /**< Output color value */\n"

        "void main(void) {\n"
        "    gl_Position = in_vertex;\n"
        "    texcoord    = in_texcoord;\n"
        "    color       = in_color;\n"
        "}\0";

    const char* fragmentShaderSource = "#version 450\n"
    "\n"
    "layout(location = 0) out vec4 out_color;\n"
    "\n"
    "layout(std430, binding = 0) buffer colorLayout\n"
    "{\n"
    "    float data[];\n"
    "} source;\n"
    "\n"
    "uniform bool flip = false;\n"
    "uniform int width = 1000;\n"
    "uniform int height = 800;\n"
    "\n"
    "in vec4 texcoord;\n"
    "\n"
    "void main(void)\n"
    "{\n"
    "    int x = int(texcoord.x * width);\n"
    "    int y;\n"
    "\n"
    "    if(flip)\n"
    "        y = height - 1 - int(texcoord.y * height);\n"
    "    else\n"
    "        y = int(texcoord.y * height);\n"
    "\n"
    "    float r = source.data[0 * width * height + (y * width + x)];\n"
    "    float g = source.data[1 * width * height + (y * width + x)];\n"
    "    float b = source.data[2 * width * height + (y * width + x)];\n"
    "    vec4 color   = vec4(r, g, b, 1);\n"
    "    out_color    = color;\n"
    "}\n";

    // Private members for the canvas
};
std::vector<Gaussian> frustum_cull(std::vector<Gaussian>& g, Camera& cam, const float& treshold);
void transformAndTileGaussians(std::vector<Gaussian>& g, std::vector<lightWeightGaussian>& out, Camera& cam,
                                TilingInformation& tile_info, float k);
