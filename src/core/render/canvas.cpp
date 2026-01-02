#include "canvas.hpp"
#include "gaussians.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "misc.cuh"
#include "render.cuh"

Canvas::Canvas(int height_, int width_, int tile_x, int tile_y): width(width_), height(height_), 
    window(nullptr, glfwDestroyWindow), d_out_pixels(height_ * width_ * 3), 
    tile_info(tile_x, tile_y, height_, width_), initial_cap(width_*height_), gaussians(nullptr), numGaussians(0), k(0.0f) {
    // Initialize canvas with tiling info
};
Canvas::~Canvas() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (this->gaussians) {
        cudaDeviceSynchronize();
        cudaFree(this->gaussians);
        this->gaussians = nullptr;
        this->numGaussians = 0;
    }
};
void Canvas::init() {
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    // only for Mac OS X
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    const int W = this->width, H = this->height;
    this->window.reset(glfwCreateWindow(W, H, "My Blank Viewer", nullptr, nullptr));
    if (!this->window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(this->window.get());
    glfwSwapInterval(1); // vsync
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return;
    }  


    glViewport(0, 0, W, H);

    // initialize vertex BUFFER OBJECT (VBO)
    // first arg is number of buffers we want
    // second arg is the address of the buffer we want to initialize
    // so if we pass >1 we need to pass an array of unsigned ints
    glGenBuffers(1, &this->VBO);  
    // we just specify to openGL this is a vertex buffer (GL_ARRAY_BUFFER)
    glBindBuffer(GL_ARRAY_BUFFER, this->VBO);

    glGenVertexArrays(1, &this->VAO);
    // 1. bind Vertex Array Object
    glBindVertexArray(this->VAO);

    // Two triangles covering NDC [-1,1], UV [0,1]
    const float U1 = 0.999999f, V1 = 0.999999f; // or std::nextafter(1.0f, 0.0f)
    this->quad[0] ={{-1,-1,0,1}, {0,0,0,0}, {1,1,1,1}};
    this->quad[1] = {{ 1,-1,0,1}, {U1,0,0,0}, {1,1,1,1}};
    this->quad[2] = {{ 1, 1,0,1}, {U1,V1,0,0}, {1,1,1,1}};
    this->quad[3] = {{-1,-1,0,1}, {0,0,0,0}, {1,1,1,1}};
    this->quad[4] = {{ 1, 1,0,1}, {U1,V1,0,0}, {1,1,1,1}};
    this->quad[5] = {{-1, 1,0,1}, {0,V1,0,0}, {1,1,1,1}};

    glBufferData(GL_ARRAY_BUFFER, sizeof(this->quad), this->quad, GL_STATIC_DRAW);

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

    

    this->vertexShader = glCreateShader(GL_VERTEX_SHADER);

    glShaderSource(this->vertexShader, 1, &this->vertexShaderSource, NULL);
    glCompileShader(this->vertexShader);

    this->fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(this->fragmentShader, 1, &this->fragmentShaderSource, NULL);
    glCompileShader(this->fragmentShader);

    
    this->shaderProgram = glCreateProgram();
    glAttachShader(this->shaderProgram, this->vertexShader);
    glAttachShader(this->shaderProgram, this->fragmentShader);
    glLinkProgram(this->shaderProgram);

    glUseProgram(this->shaderProgram);


    
    glGenBuffers(1, &this->ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->ssbo);

    glBufferData(GL_SHADER_STORAGE_BUFFER, W*H*3*sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this->ssbo); // <-- binding = 0 matches the shader

    // 5) Set required uniforms (don’t rely on defaults)
    glUniform1i(glGetUniformLocation(this->shaderProgram, "width"),  W);
    glUniform1i(glGetUniformLocation(this->shaderProgram, "height"), H);
    glUniform1i(glGetUniformLocation(this->shaderProgram, "flip"),   GL_FALSE); // or GL_TRUE
    glfwSetWindowUserPointer(this->window.get(), this);

    glfwSetMouseButtonCallback(this->window.get(), [](GLFWwindow* window, int button, int action, int mods) {
        Canvas* canvas = static_cast<Canvas*>(glfwGetWindowUserPointer(window));
        if (!canvas) {
            // throw error
            std::cerr << "Error: Canvas pointer is null in MouseCallback\n";
            return;
        }
        canvas->MouseCallback(window, button, action, mods);
    });
    glfwSetCursorPosCallback(this->window.get(), [](GLFWwindow* window, double xpos, double ypos) {
        Canvas* canvas = static_cast<Canvas*>(glfwGetWindowUserPointer(window));
        if (!canvas) {
            // throw error
            std::cerr << "Error: Canvas pointer is null in MouseCallback\n";
            return;
        }
        canvas->CursorPosCallback(window, xpos, ypos);
    });

    glfwSetFramebufferSizeCallback(window.get(),
        [](GLFWwindow* win, int fbW, int fbH) {
            glViewport(0, 0, fbW, fbH);

            // Get your Canvas instance back
            auto* canvas = static_cast<Canvas*>(glfwGetWindowUserPointer(win));
            if (!canvas) return;

            canvas->onResize(fbW, fbH);
        }
    );
    glfwSetWindowUserPointer(window.get(), this);

    glfwSetScrollCallback(this->window.get(), [](GLFWwindow* window, double xoffset, double yoffset) {
        Canvas* canvas = static_cast<Canvas*>(glfwGetWindowUserPointer(window));
        if (!canvas) {
            std::cerr << "Error: Canvas pointer is null in ScrollCallback\n";
            return;
        }
        canvas->ScrollCallback(window, xoffset, yoffset);
    });

    glfwSetDropCallback(this->window.get(), [](GLFWwindow* win, int count, const char** paths) {
        Canvas* canvas = static_cast<Canvas*>(glfwGetWindowUserPointer(win));
        if (!canvas) {
            std::cerr << "Error: Canvas pointer is null in ScrollCallback\n";
            return;
        }
        canvas->dropFileCallback(win, count, paths);
    });

    //setup imgui
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io= ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(this->window.get(), false);          // Second param install_callback=true will install GLFW callbacks and chain to existing ones.
    ImGui_ImplOpenGL3_Init();
    
    this->settings.num_tile_x = this->tile_info.num_tile_x;
    this->settings.num_tile_y = this->tile_info.num_tile_y;
    this->settings.fov = this->cam->fovY;
    return;
};

void Canvas::onResize(int fbW, int fbH) {
    width  = fbW;
    height = fbH;

    cam->setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
    cam->updateCameraMatrices();

    this->d_out_pixels.resize(width * height * 3);
    this->tile_info.resize(height,width, this->tile_info.num_tile_x, this->tile_info.num_tile_y);

    if ((fbW * fbH) > this->initial_cap) {
        // handle reallocating ssbo
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, fbW*fbH*3*sizeof(float), nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this->ssbo);
        this->initial_cap = fbW * fbH;
    }
    // update new viewport
    glViewport(0, 0, fbW, fbH);

    //  update fragment shaders index
    glUseProgram(this->shaderProgram);
    glUniform1i(glGetUniformLocation(this->shaderProgram, "width"),  fbW);
    glUniform1i(glGetUniformLocation(this->shaderProgram, "height"), fbH);
    glUniform1i(glGetUniformLocation(this->shaderProgram, "flip"),   GL_FALSE);

}

void Canvas::CursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);
    if (ImGui::GetIO().WantCaptureMouse) {
        this->controls.dragging = false;
        return;
    }
    // Handle cursor position callback
    if (!this->controls.dragging) return;
    if (!this->cam) return;   // no camera attached → nothing to orbit

    double dx = xpos - this->controls.lastX;
    double dy = ypos - this->controls.lastY;
    this->controls.lastX = xpos;
    this->controls.lastY = ypos;

    // Convert pixels to degrees of orbit
    float dAzimuth   = static_cast<float>(-dx) * this->controls.orbitSpeedX;
    float dElevation = static_cast<float>(dy) * this->controls.orbitSpeedY;  // invert Y so drag up = look down, or flip if you prefer

    // This orbits RELATIVE to current azimuth/elevation
    this->cam->orbit(dAzimuth, dElevation);
}
void Canvas::ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (!this->cam) return;

    // yoffset > 0 : scroll up
    // yoffset < 0 : scroll down
    float zoomDelta = static_cast<float>(yoffset) * this->controls.zoomSpeed;

    // Convention: positive delta = zoom in (or out) depending on your Camera::zoom
    this->cam->zoom(zoomDelta);

    // Optional debug:
    // std::cout << "Zoom scroll yoffset=" << yoffset
    //           << " -> zoomDelta=" << zoomDelta << "\n";
}
void Canvas::MouseCallback(GLFWwindow* window, int button, int action, int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    if (ImGui::GetIO().WantCaptureMouse) {
        if (action == GLFW_PRESS) this->controls.dragging = false;
        return;
    }
    // Handle mouse button callback
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            this->controls.dragging = true;
            glfwGetCursorPos(window, &this->controls.lastX, &this->controls.lastY);
        } else if (action == GLFW_RELEASE) {
            this->controls.dragging = false;
        }
    }
}
void Canvas::dropFileCallback(GLFWwindow* win, int count, const char** paths) {
    int indx = count-1; // we will use the last path
    this->loadGaussians(paths[indx]);
};

void Canvas::loadGaussians(const std::string& filename) {
    if (this->gaussians) {
        cudaDeviceSynchronize();
        cudaFree(this->gaussians);
        this->gaussians = nullptr;
        this->numGaussians = 0;
    }
    this->gaussians = loadGaussianCudaFromPly(filename, &this->numGaussians);
    if (this->gaussians == nullptr) {
        std::cerr << "Failed to load gaussians from PLY file: " << filename << std::endl;
    }
};

void Canvas::debugWindow() {
    ImGui::SetNextWindowSize(ImVec2(500, 500), ImGuiCond_FirstUseEver); //ImGuiCond_Always);
    ImGui::Begin("Settings", &settings.show_settings);


    // Renderer-ish
    ImGui::Checkbox("Flip Y", &settings.flip);
    ImGui::SliderFloat("k-sigma (splat radius)", &this->k, 0.1f, 8.0f, "%.2f");


    // Controls

    if (ImGui::SliderFloat("fovY", &settings.fov, 75.0f, 120.0f, "%.2f")) {
        this->cam->setFovY(settings.fov);
        this->cam->updateCameraMatrices();
        this->cam->updateFrustumPlanes();
    };


    ImGui::Checkbox("Lock X/Y tiles", &settings.lock_tiles);

    bool x_changed = ImGui::SliderInt("X tiles", &settings.num_tile_x, 40, 64);

        
    bool y_changed = false;
    ImGui::BeginDisabled(settings.lock_tiles);
    y_changed = ImGui::SliderInt("Y tiles", &settings.num_tile_y, 40, 64);
    ImGui::EndDisabled();

    if (settings.lock_tiles && x_changed) {
        settings.num_tile_y = settings.num_tile_x;
    }
    if ((x_changed || y_changed)) {
        this->tile_info.resize(tile_info.H, tile_info.W, settings.num_tile_x, settings.num_tile_y);
    }

    ImGui::End();
};

void Canvas::render() {
    preprocessCUDAGaussians(this->gaussians, this->d_out_pixels.data(), this->numGaussians, *(this->cam), 
                this->tile_info.num_tile_y, this->tile_info.num_tile_x, this->tile_info.width_stride,
                this->tile_info.height_stride, this->tile_info.W, this->tile_info.H, k);
    this->draw(this->d_out_pixels.data());
};

void Canvas::draw(float* pixel_out) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    this->debugWindow();
    // Draw the contents of pixel_out to the screen
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, this->width*this->height*3*sizeof(float), pixel_out);


    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(this->shaderProgram);
    glBindVertexArray(this->VAO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this->ssbo);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(this->window.get());
    glfwPollEvents();
}
