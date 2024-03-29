#ifndef BUFFER_H
#define BUFFER_H

#include <vulkan/vulkan.h>

namespace Buffer {

    VkCommandBuffer recordCopyCommand(
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkBuffer srcBuffer,
        VkBuffer dstBuffer,
        VkDeviceSize size,
        size_t srcOffset,
        size_t dstOffset);

    VkCommandBuffer recordCopyCommand(
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkBuffer srcBuffer,
        VkBuffer dstBuffer,
        VkDeviceSize size);

    void copyBuffer(
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkQueue queue,
        VkBuffer& srcBuffer,
        VkBuffer& dstBuffer,
        VkDeviceSize size,
        size_t srcOffset,
        size_t dstOffset);

    void copyBuffer(
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkQueue queue,
        VkBuffer& srcBuffer,
        VkBuffer& dstBuffer,
        VkDeviceSize size);

    void createBuffer(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkBuffer& buffer,
        VkDeviceMemory& bufferMemory);

    void createBufferWithData(
        void* data,
        VkDeviceSize bufferSize,
        VkBufferUsageFlags usage,
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkQueue queue,
        VkBuffer& buffer,
        VkDeviceMemory& bufferMemory);

    void writeHostVisible(
        void* src,
        VkDeviceMemory dst,
        VkDeviceSize offset,
        VkDeviceSize size,
        VkDevice logicalDevice);

    void readHostVisible(
        VkDeviceMemory src,
        void* dst,
        VkDeviceSize offset,
        VkDeviceSize size,
        VkDevice logicalDevice);

    void copyDeviceBufferToHost(
        void* data,
        VkDeviceSize bufferSize,
        VkBuffer buffer,
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkQueue queue);

    void copyHostToDeviceBuffer(
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
