#include <Renderer/Surface.h>

#include <Renderer/Constants.h>

#include <iostream>

namespace {

    static void frameBufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<Surface::Window*>(glfwGetWindowUserPointer(window));
        app->m_hasBeenResized = true;
    }

} // namespace anonymous

std::shared_ptr<Surface::Window> Surface::createWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow* window = glfwCreateWindow(Constants::kWidth, Constants::kHeight, "Vulkan Game", nullptr, nullptr);

    auto windowAndResizeFlag = std::shared_ptr<Window>(new Window{window, false});

    glfwSetWindowUserPointer(window, windowAndResizeFlag.get());
    glfwSetFramebufferSizeCallback(window, frameBufferResizeCallback);

    return windowAndResizeFlag;
}

VkSurfaceKHR Surface::createSurface(VkInstance instance, GLFWwindow* window) {

    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface");
    }
    return surface;
}
