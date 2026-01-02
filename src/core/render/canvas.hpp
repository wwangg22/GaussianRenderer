#pragma once
#include "render.hpp"
#include <glad/glad.h> 
#include <GLFW/glfw3.h>
#include "gaussians.hpp"

struct UiSettings {
    bool show_settings = true;
    bool flip = false;
    float k_sigma = 3.0f;          // your 'k' for radius
    float exposure = 1.0f;         // example post-param
    float zoomSpeed = 0.10f;
    float orbitSpeedX = 0.25f;
    float orbitSpeedY = 0.25f;
    int num_tile_x = 40;
    int num_tile_y = 40;
    bool lock_tiles = true;
    float fov = 90.0f;
};


class Canvas {
public: 
    int width;
    int height;
    int initial_cap;
    using windowPtr = std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)>;
    windowPtr window;
    Camera* cam;
    OrbitControls controls;
    std::vector<float> d_out_pixels;
    TilingInformation tile_info;

 
    Canvas( int height_, int width_, int tile_x, int tile_y);
    ~Canvas();
    void init();
    void draw(float* pixel_out);
    void onResize(int fbW, int fbH);
    void loadGaussians(const std::string& filename);
    void MouseCallback(GLFWwindow* window, int button, int action, int mods);
    void CursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    void dropFileCallback(GLFWwindow* win, int count, const char** paths);
    void render();

private:
    Gaussian * gaussians;
    int numGaussians;
    unsigned int VBO;
    unsigned int VAO;

    float k;

    Vertex quad[6];
    unsigned int vertexShader;
    unsigned int fragmentShader;
    unsigned int shaderProgram;

    UiSettings settings{};
    void debugWindow();

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