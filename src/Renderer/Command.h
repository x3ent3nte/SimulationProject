#ifndef COMMAND_H
#define COMMAND_H

#include <vulkan/vulkan.h>

#include <vector>

namespace Command {

    VkCommandPool createCommandPool(
        VkDevice logicalDevice,
        uint32_t queueIndex);

    VkCommandBuffer beginSingleTimeCommands(
        VkDevice logicalDevice,
        VkCommandPool commandPool);

    void endSingleTimeCommands(
        VkCommandBuffer commandBuffer,
        VkQueue queue,
        VkDevice logicalDevice,
        VkCommandPool commandPool);
}

#endif
