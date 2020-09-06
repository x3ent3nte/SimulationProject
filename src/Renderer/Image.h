#ifndef IMAGE_H
#define IMAGE_H

#include <vulkan/vulkan.h>

namespace Image {

    void createImage(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        uint32_t width,
        uint32_t height,
        uint32_t mipLevels,
        VkSampleCountFlagBits numSamples,
        VkFormat format,
        VkImageTiling tiling,
        VkImageUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkImage& image,
        VkDeviceMemory& imageMemory);

    VkImageView createImageView(
        VkDevice logicalDevice,
        VkImage image,
        VkFormat format,
        VkImageAspectFlags aspectFlags,
        uint32_t mipLevels);

    uint32_t createTextureImage(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkQueue queue,
        VkImage& textureImage,
        VkDeviceMemory& textureImageMemory);

    void transitionImageLayout(
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkQueue queue,
        VkImage image,
        VkFormat format,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        uint32_t mipLevels);

    VkSampler createTextureSampler(VkDevice logicalDevice, uint32_t mipLevels);
}

#endif
