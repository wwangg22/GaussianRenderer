#include <GLFW/glfw3.h>
#include <iostream>

int window() {
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return 1;
    }

    // Request an OpenGL 3.3 core context (works fine for simple clearing, too)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
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

    // Basic event: close on ESC
    glfwSetKeyCallback(win, [](GLFWwindow* w, int key, int, int action, int) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(w, 1);
    });

    // Main loop
    while (!glfwWindowShouldClose(win)) {
        // Clear to dark gray
        glClearColor(0.12f, 0.12f, 0.13f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}