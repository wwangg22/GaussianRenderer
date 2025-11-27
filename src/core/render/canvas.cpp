#include "canvas.hpp"


Canvas::Canvas(int height, int width) {
    // Initialize canvas with tiling info
    this->width = width;
    this->height = height;
}
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
    this->window = glfwCreateWindow(W, H, "My Blank Viewer", nullptr, nullptr);
    if (!this->window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(this->window);
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
    glfwSetWindowUserPointer(this->window, this);

    glfwSetMouseButtonCallback(this->window, [](GLFWwindow* window, int button, int action, int mods) {
        Canvas* canvas = static_cast<Canvas*>(glfwGetWindowUserPointer(window));
        if (!canvas) {
            // throw error
            std::cerr << "Error: Canvas pointer is null in MouseCallback\n";
            return;
        }
        canvas->MouseCallback(window, button, action, mods);
    });
    glfwSetCursorPosCallback(this->window, [](GLFWwindow* window, double xpos, double ypos) {
        Canvas* canvas = static_cast<Canvas*>(glfwGetWindowUserPointer(window));
        if (!canvas) {
            // throw error
            std::cerr << "Error: Canvas pointer is null in MouseCallback\n";
            return;
        }
        canvas->CursorPosCallback(window, xpos, ypos);
    });

    glfwSetScrollCallback(this->window, [](GLFWwindow* window, double xoffset, double yoffset) {
        Canvas* canvas = static_cast<Canvas*>(glfwGetWindowUserPointer(window));
        if (!canvas) {
            std::cerr << "Error: Canvas pointer is null in ScrollCallback\n";
            return;
        }
        canvas->ScrollCallback(window, xoffset, yoffset);
    });
    return;
};

void Canvas::CursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
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

    std::cout << "Orbiting camera by (" << dAzimuth << ", " << dElevation << ")\n";
    // This orbits RELATIVE to current azimuth/elevation
    this->cam->orbit(dAzimuth, dElevation);
}
void Canvas::ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
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



void Canvas::draw(float* pixel_out) {
// Draw the contents of pixel_out to the screen
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, this->width*this->height*3*sizeof(float), pixel_out);


    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(this->shaderProgram);
    glBindVertexArray(this->VAO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this->ssbo);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glfwSwapBuffers(this->window);
    glfwPollEvents();
}
