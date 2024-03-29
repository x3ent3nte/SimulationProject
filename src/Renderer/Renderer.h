#ifndef RENDERER_H
#define RENDERER_H

#include <Renderer/Surface.h>
#include <Renderer/Connector.h>
#include <Renderer/Mesh.h>
#include <Renderer/Texture.h>

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
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue graphicsQueue,
        VkQueue presentQueue,
        VkCommandPool commandPool,
        std::shared_ptr<Connector> connector,
        std::shared_ptr<Mesh> mesh,
        const std::vector<std::shared_ptr<Texture>>& textures,
        uint32_t maxNumberOfAgents);
};

#endif
