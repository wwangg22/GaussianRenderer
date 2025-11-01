#include <chrono>
#include <cuda_runtime.h>
#include <glad/glad.h> 
#include <GLFW/glfw3.h>

#include "render.cuh"
#include "gaussians.hpp"
#include "render.hpp"
#include "camera.hpp"


int main(int argc, char* argv[]){
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_ply_file>\n";
        return 1;
    }
    float p[3] = {20.0f, 20.0f, 20.0f};
    Camera cam;
    cam.setFovY(90.0f);
    cam.setAspectRatio(800.0f / 600.0f);
    cam.setClippingPlanes(0.1f, 100.0f);
    cam.setPosition(p);
    cam.updateCameraMatrices();
    cam.updateFrustumPlanes();

    std::string filename = argv[1];
    std::vector<Gaussian> gaussians = loadGaussiansFromPly(filename);

    std::cout << "Loaded " << gaussians.size() << " gaussians from " << filename << std::endl;

    // start timing
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Gaussian> culled_gaussians = frustum_cull(gaussians, cam, 0.01f);
    // end timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "After culling: " << culled_gaussians.size() << " gaussians remain." << std::endl;
    std::cout << "Culling took " << elapsed.count() << " ms." << std::endl;

    std::vector<lightWeightGaussian> tiled_gaussians;
    TilingInformation tile_info(16, 16, 800, 600);
    transformAndTileGaussians(culled_gaussians, tiled_gaussians, cam, tile_info, 3.0f);
    std::cout << "Tiled " << tiled_gaussians.size() << " gaussians." << std::endl;
    float kernel_ms;
    oneSweep3DGaussianSort(tiled_gaussians.data(), 
                          tiled_gaussians.size(), 
                          64, // using 64 bits for radix sort
                          &kernel_ms);

    std::cout << "Kernel execution time: " << kernel_ms << " ms." << std::endl;

    float *d_out_pixels = (float*) malloc(800 * 600 * 3 * sizeof(float));


    renderGaussiansCUDA(d_out_pixels, &tile_info, culled_gaussians.data(), tiled_gaussians.data(), culled_gaussians.size(), tiled_gaussians.size(), &kernel_ms);

    // print render time
    std::cout << "Rendering execution time: " << kernel_ms << " ms." << std::endl;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    // only for Mac OS X
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    const int W = 800, H = 600;
    GLFWwindow* win = glfwCreateWindow(W, H, "My Blank Viewer", nullptr, nullptr);
    if (!win) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(win);
    glfwSwapInterval(1); // vsync
     if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return 1;
    }  
    glViewport(0, 0, W, H);

    // initialize vertex BUFFER OBJECT (VBO)
    unsigned int VBO;
    // first arg is number of buffers we want
    // second arg is the address of the buffer we want to initialize
    // so if we pass >1 we need to pass an array of unsigned ints
    glGenBuffers(1, &VBO);  
    // we just specify to openGL this is a vertex buffer (GL_ARRAY_BUFFER)
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    // 1. bind Vertex Array Object
    glBindVertexArray(VAO);

    // 1) Create a fullscreen QUAD with positions + texcoords + color
    struct Vertex {
        float pos[4];     // in_vertex
        float uv[4];      // in_texcoord (z,w unused)
        float col[4];     // in_color
    };

    // Two triangles covering NDC [-1,1], UV [0,1]
    const float U1 = 0.999999f, V1 = 0.999999f; // or std::nextafter(1.0f, 0.0f)
    Vertex quad[6] = {
    {{-1,-1,0,1}, {0,0,0,0}, {1,1,1,1}},
    {{ 1,-1,0,1}, {U1,0,0,0}, {1,1,1,1}},
    {{ 1, 1,0,1}, {U1,V1,0,0}, {1,1,1,1}},
    {{-1,-1,0,1}, {0,0,0,0}, {1,1,1,1}},
    {{ 1, 1,0,1}, {U1,V1,0,0}, {1,1,1,1}},
    {{-1, 1,0,1}, {0,V1,0,0}, {1,1,1,1}},
    };


    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

    GLsizei stride = sizeof(Vertex);
    std::size_t off_pos = offsetof(Vertex, pos);
    std::size_t off_uv  = offsetof(Vertex, uv);
    std::size_t off_col = offsetof(Vertex, col);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, stride, (void*)off_pos);

    // attrib 1: in_texcoord (vec4) - we use uv in xy
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, (void*)off_uv);

    // attrib 2: in_color (vec4) - constant white here
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, (void*)off_col);

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
    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);

    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    unsigned int shaderProgram;
    shaderProgram = glCreateProgram();

    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glUseProgram(shaderProgram);


    GLuint ssbo=0;
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, W*H*3*sizeof(float), d_out_pixels, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo); // <-- binding = 0 matches the shader

    // 5) Set required uniforms (don’t rely on defaults)
    glUniform1i(glGetUniformLocation(shaderProgram, "width"),  W);
    glUniform1i(glGetUniformLocation(shaderProgram, "height"), H);
    glUniform1i(glGetUniformLocation(shaderProgram, "flip"),   GL_FALSE); // or GL_TRUE



    while (!glfwWindowShouldClose(win)) {
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(win);

    }


    return 0;
}