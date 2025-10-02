#include <glad/glad.h> 
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <array>
#include "window.hpp"
#include "camera.hpp"
#include "math.hpp"


struct AppState {
    Camera* cam = nullptr;
    bool dragging = false;
    double lastX = 0.0, lastY = 0.0;
    float orbitSensitivity = 0.5f; // tweak to taste (radians per pixel)
};
 // callback for resizing window
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    if (height == 0) return;
    auto* cam = reinterpret_cast<Camera*>(glfwGetWindowUserPointer(window));
    if (cam) {
        cam->setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
        cam->updateCameraMatrices();
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    std::cout << "Scroll delta: (" << xoffset << ", " << yoffset << ")\n";
    // Retrieve the AppState* we stored on the window
    auto* state = reinterpret_cast<AppState*>(glfwGetWindowUserPointer(window));
    if (!state || !state->cam) return;

    // Pass the scroll delta to your camera. Many UIs prefer only yoffset.
    // Feel free to add a sensitivity scalar here if you like (e.g., 0.1f * yoffset).
    state->cam->zoom(static_cast<float>(yoffset));

    // // If zoom() changes the projection/fov, make sure matrices are refreshed
    // cam->updateCameraMatrices();
}


void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    auto* state = reinterpret_cast<AppState*>(glfwGetWindowUserPointer(window));
    if (!state || !state->cam) return;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            state->dragging = true;
            glfwGetCursorPos(window, &state->lastX, &state->lastY);
        } else if (action == GLFW_RELEASE) {
            state->dragging = false;
        }
    }
}

void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
    auto* state = reinterpret_cast<AppState*>(glfwGetWindowUserPointer(window));
    if (!state || !state->cam || !state->dragging) return;

    double dx = xpos - state->lastX;
    double dy = ypos - state->lastY;
    state->lastX = xpos;
    state->lastY = ypos;

    // Map pixels → angle deltas
    float dazimuth   = static_cast<float>( dx) * state->orbitSensitivity; // right drag → +azimuth
    float delevation = static_cast<float>( dy) * state->orbitSensitivity; // down  drag → +elevation (your request)

    state->cam->orbit(dazimuth, delevation);
    std::cout << "Orbit delta (radians): (" << dazimuth << ", " << delevation << ")\n";
    state->cam->updateCameraMatrices();
}
int window(int argc, char** argv) {
    // sets up internals of glfw, like figure out what OS we are on etc
    // must be called at the start of the program
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return 1;
    }
    if (argc != 2){
        std::cout << "Usage: " << argv[0] << " float \n";
        return 1;
    }
    float p[3] = {0.0f, 1.0f, 3.0f};
    Camera cam;
    cam.setFovY(std::atof(argv[1]));
    cam.setAspectRatio(800.0f / 600.0f);
    cam.setClippingPlanes(0.1f, 100.0f);
    cam.setPosition(p);
    cam.updateCameraMatrices();
    
    std::vector<std::array<float, 3>> vec;
    std::vector<float> points;

    vec.push_back({0.0f, 0.0f, 0.0f});
    vec.push_back({1.0f, 0.0f, 0.0f});
    vec.push_back({0.0f, 1.0f, 0.0f});
    vec.push_back({0.0f, 0.0f, 1.0f});
    //print out points before trans
    std::cout << "Original Points:\n";
    for (const auto& v : vec) {
        std::cout << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")\n";
    }
    // print out Vmatrix
    std::cout << "View Matrix:\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << cam.V_matrix[i * 4 + j] << " ";
        }
        std::cout << "\n";
    }

    //print out projection matrix
    std::cout << "Projection Matrix:\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << cam.P_matrix[i * 4 + j] << " ";
        }
        std::cout << "\n";
    }

    //print out M matrix
    std::cout << "M Matrix:\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << cam.M_matrix[i * 4 + j] << " ";
        }
        std::cout << "\n";
    }

    for (const auto& v : vec) {
        float point[4] = {v[0], v[1], v[2], 1.0f};
        float out[3];
        cam.transformPointToCameraSpace(point, out);
        points.push_back(out[0]);
        points.push_back(out[1]);
        points.push_back(out[2]);
        // std::cout << "Point in screen space: (" << out[0] << ", " << out[1] << ", " << out[2] << ", " << out[3] << ")\n";
    }

    std::cout << "Camera Position: (" << cam.position[0] << ", " << cam.position[1] << ", " << cam.position[2] << ")\n";

    // basically these functions tells GLFW what version of OpenGL we want
    // once we follow up with glfwCreateWindow to get the actual window
    // more settings that can be set found here: https://www.glfw.org/docs/latest/window.html#window_hints
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
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
    // initialize GLAD
    // GLAD manages function pointers
    // basically functions like glClearColor are initialized as NULL ptrs before 
    // you "initialize" GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return 1;
    }    
    // sets the size of the rendering window
    // first two args are the lower left corner of the window
    // last two args are the width and height
    glViewport(0, 0, W, H);

    // initialize vertex BUFFER OBJECT (VBO)
    unsigned int VBO;
    // first arg is number of buffers we want
    // second arg is the address of the buffer we want to initialize
    // so if we pass >1 we need to pass an array of unsigned ints
    glGenBuffers(1, &VBO);  
    // we just specify to openGL this is a vertex buffer (GL_ARRAY_BUFFER)
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // copies the actual data to the GPU
    // second arg is the size of the data in bytes
    // third arg is the actual pointer to the data
    // last arg is how we want the GPU to manage the data
    // GL_STATIC_DRAW: data will most likely not change at all or very rarely
    // GL_DYNAMIC_DRAW: data is likely to change a lot
    // GL_STREAM_DRAW: data will change every time it is drawn
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * 4, points.data(), GL_STREAM_DRAW);

    //shaders code raw string
    // ok the layout (location = 0) means this is the 0th attribute
    // we will need this once we tell the vertex buffer how the data is structured
    const char* vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
    "}\0";
    // here is where we create the shaders
    unsigned int vertexShader;
    // pass what type of shaders we want (vertex, fragment, geometry, etc))
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    // first the shader object id
    // second arg is the number of strings (1 if single string)
    // third arg is the actual string of code
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    //so to check if the shaders compiled successfully
    int  success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);

    if(!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    const char* fragmentShaderSource = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "void main()\n"
    "{\n"
    "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
    "}\n\0";

    // process of creating fragment shader is same as vertex shader
    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }


    // now we just need a shader program 
    unsigned int shaderProgram;
    //creates program and returns the ID
    shaderProgram = glCreateProgram();

    //attach shaders to program
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // we can also check if linking was successful
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    // now we can use it by calling
    glUseProgram(shaderProgram);
    
    // so we told our vertex buffer how big the buffer was
    // but we still need to tell it how the data is structured
    // this is done through vertex attributes
    // the first arg (0) is the same as the layout (location = 0) in the vertex shader
    // second arg (3) is how many components there are for each vertex
    // third arg is the type of each component
    // fourth arg is if the data is normalized (i.e. GL_FALSE)
    // fifth arg is the stride (space between consecutive vertex attributes) 
    //      if tightly packed just put 0 and OpenGL figures it out
    // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    // first arg is which vertex attribute we want to configure
    // glEnableVertexAttribArray(0);
    // right now we commented this out cuz we have VAO doing it later

    // so normally you would have to draw the data like this: (without VAO)
    // // 0. copy our vertices array in a buffer for OpenGL to use
    // glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // // 1. then set the vertex attributes pointers
    // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    // glEnableVertexAttribArray(0);  
    // // 2. use our shader program when we want to render an object
    // glUseProgram(shaderProgram);
    // // 3. now draw the object 
    // someOpenGLFunctionThatDrawsOurTriangle(); 

    //initialization for VAO is similar to VBO
    unsigned int VAO;
    glGenVertexArrays(1, &VAO);

    // 1. bind Vertex Array Object
    glBindVertexArray(VAO);
    // 2. copy our vertices array in a buffer for OpenGL to use
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), points.data(), GL_DYNAMIC_DRAW);
    // 3. then set our vertex attributes pointers
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);  


    static AppState app;              // static or keep it alive as long as the window
    app.cam = &cam;

    //tell glfw to use this function on every window resize
    glfwSetFramebufferSizeCallback(win, framebuffer_size_callback);

    // Basic event: close on ESC
    glfwSetKeyCallback(win, [](GLFWwindow* w, int key, int, int action, int) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(w, 1);
    });
    // attach camera to window for access in callbacks
    glfwSetWindowUserPointer(win, &app);

    glfwSetScrollCallback(win, scroll_callback);

    glfwSetMouseButtonCallback(win, mouse_button_callback);
    glfwSetCursorPosCallback(win, cursor_pos_callback);

    // Main loop
    // glfwWindowShouldClose checks if the window was instructed to close (e.g. by pressing the 'X' button)
    while (!glfwWindowShouldClose(win)) {
        for (int idx = 0; idx < points.size()/3; idx++) {
            const auto& v = vec[idx];
            float point[4] = {v[0], v[1], v[2], 1.0f};
            float out[4];
            cam.transformPointToCameraSpace(point, out);
            points[3*idx+0] = out[0];
            points[3*idx+1] = out[1];
            points[3*idx+2] = out[2];
            // std::cout << "Point in screen space: (" << out[0] << ", " << out[1] << ", " << out[2] << ", " << out[3] << ")\n";
        }

        glBufferSubData(GL_ARRAY_BUFFER, 0, points.size()*sizeof(float), points.data());
        // Clear to dark gray
        glClearColor(0.12f, 0.12f, 0.13f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        // draw our first triangle
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // swap color buffer (large 2D buffer that contains color values for each pixel)
        // double buffer in practice to avoid flickering
        // we draw to the back buffer while the front buffer is being displayed
        glfwSwapBuffers(win);
        // checks if any events are triggered (like keyboard input or mouse movement)
        glfwPollEvents();
    }
    //clean up shaders
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    //clean up for glfw
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}