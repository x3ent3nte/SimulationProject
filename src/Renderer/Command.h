#ifndef COMMAND_H
#define COMMAND_H

#include <vulkan/vulkan.h>

#include <vector>

namespace Command {

    VkCommandPool createCommandPool(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkSurfaceKHR surface);

    VkCommandBuffer beginSingleTimeCommands(
        VkDevice logicalDevice,
        VkCommandPool commandPool);

    void endSingleTimeCommands(
        VkCommandBuffer commandBuffer,
        VkQueue queue,
        VkDevice logicalDevice,
        VkCommandPool commandPool);

    void createCommandBuffers(
        const std::vector<VkFramebuffer>& swapChainFrameBuffers,
        VkCommandPool commandPool,
        VkDevice logicalDevice,
        VkRenderPass renderPass,
        VkExtent2D swapChainExtent,
        const std::vector<VkBuffer>& instanceBuffers,
        VkBuffer vertexBuffer,
        VkBuffer indexBuffer,
        uint32_t numIndices,
        uint32_t numInstances,
        const std::vector<VkDescriptorSet>& descriptorSets,
        VkPipeline graphicsPipeline,
        VkPipelineLayout pipelineLayout,
        std::vector<VkCommandBuffer>& commandBuffers);
}

#endif
