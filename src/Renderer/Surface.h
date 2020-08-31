#ifndef SURFACE_H
#define SURFACE_H

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <memory>
#include <stdexcept>

namespace Surface {

    struct Window {
        GLFWwindow* m_window;
        bool m_hasBeenResized;
    };

    std::shared_ptr<Window> createWindow();

    VkSurfaceKHR createSurface(VkInstance instance, GLFWwindow* window);
}

#endif
