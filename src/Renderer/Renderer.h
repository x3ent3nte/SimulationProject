#ifndef RENDERER_H
#define RENDERER_H

#include <Renderer/KeyboardControl.h>
#include <Renderer/Surface.h>
#include <Renderer/Connector.h>
#include <Renderer/Model.h>

#include <vulkan/vulkan.h>

#include <memory>
#include <vector>

class Renderer {
public:

    virtual ~Renderer() = default;

    virtual void render(float timeDelta) = 0;

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
        std::shared_ptr<Connector> connector,
        const std::vector<std::shared_ptr<Model>>& models,
        uint32_t maxNumberOfAgents);
};

#endif
