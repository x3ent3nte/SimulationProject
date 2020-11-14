#ifndef RENDERER_H
#define RENDERER_H

#include <Renderer/KeyboardControl.h>
#include <Renderer/Surface.h>

#include <vulkan/vulkan.h>

#include <memory>

class Renderer {
public:
    int render(
        VkInstance instance,
        std::shared_ptr<Surface::Window> window,
        VkSurfaceKHR surface,
        std::shared_ptr<KeyboardControl> keyboardControl);
};

#endif
