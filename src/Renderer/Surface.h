#ifndef SURFACE_H
#define SURFACE_H

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <Renderer/KeyboardControl.h>

#include <memory>
#include <stdexcept>

namespace Surface {

    struct Window {
        GLFWwindow* m_window;
        bool m_hasBeenResized;
        std::shared_ptr<KeyboardControl> m_keyboardControl;

        void keyboardActivity(int key, int scancode, int action, int mods);
    };

    std::shared_ptr<Window> createWindow(std::shared_ptr<KeyboardControl> keyboardControl);

    VkSurfaceKHR createSurface(VkInstance instance, GLFWwindow* window);
}

#endif
