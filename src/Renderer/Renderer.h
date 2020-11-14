#ifndef RENDERER_H
#define RENDERER_H

#include <Renderer/KeyboardControl.h>
#include <Renderer/Surface.h>
#include <Renderer/Connector.h>

#include <vulkan/vulkan.h>

#include <memory>

class Renderer {
public:

    virtual ~Renderer() = default;

    virtual void render(float time) = 0;

    static std::shared_ptr<Renderer> create(
        VkInstance instance,
        std::shared_ptr<Surface::Window> window,
        VkSurfaceKHR surface,
        std::shared_ptr<KeyboardControl> keyboardControl,
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue graphicsQueue,
        VkQueue presentQueue,
        VkCommandPool commandPool,
        std::shared_ptr<Connector> connector);
};

#endif
