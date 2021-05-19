#ifndef TEXTURE_H
#define TEXTURE_H

#include <vulkan/vulkan.h>

#include <string>

class Texture {

public:

    Texture(
        const std::string& texturePath,
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool);

    virtual ~Texture();

    VkImageView m_imageView;
    VkSampler m_sampler;

    uint32_t mipLevels() const;

private:

    VkDevice m_logicalDevice;

    VkImage m_image;
    VkDeviceMemory m_imageDeviceMemory;

    uint32_t m_mipLevels;
};

#endif
