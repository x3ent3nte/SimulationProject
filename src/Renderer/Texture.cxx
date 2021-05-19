#include <Renderer/Texture.h>

#include <Renderer/Image.h>

Texture::Texture(
    const std::string& texturePath,
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool) {

    m_logicalDevice = logicalDevice;

    m_mipLevels = Image::createTextureImage(
        physicalDevice,
        m_logicalDevice,
        commandPool,
        queue,
        texturePath,
        m_image,
        m_imageDeviceMemory);

    m_imageView = Image::createImageView(m_logicalDevice, m_image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, m_mipLevels);
    m_sampler = Image::createTextureSampler(m_logicalDevice, m_mipLevels);
}

Texture::~Texture() {
    vkDestroyImage(m_logicalDevice, m_image, nullptr);
    vkFreeMemory(m_logicalDevice, m_imageDeviceMemory, nullptr);
    vkDestroyImageView(m_logicalDevice, m_imageView, nullptr);
    vkDestroySampler(m_logicalDevice, m_sampler, nullptr);
}

uint32_t Texture::mipLevels() const {
    return m_mipLevels;
}
