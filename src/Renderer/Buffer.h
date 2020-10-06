#ifndef BUFFER_H
#define BUFFER_H

#include <vulkan/vulkan.h>

namespace Buffer {
    void createBuffer(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkBuffer& buffer,
        VkDeviceMemory& bufferMemory);

    void createReadOnlyBuffer(
        void* data,
        VkDeviceSize bufferSize,
        VkBufferUsageFlags usage,
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkQueue queue,
        VkBuffer& buffer,
        VkDeviceMemory& bufferMemory);

    void copyDeviceBufferToHost(
        void* data,
        VkDeviceSize bufferSize,
        VkBuffer buffer,
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkQueue queue);

    void copyBufferToImage(
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkQueue queue,
        VkBuffer buffer,
        VkImage image,
        uint32_t width,
        uint32_t height);
}

#endif
