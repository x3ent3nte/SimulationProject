#ifndef DESCRIPTORS_H
#define DESCRIPTORS_H

#include <vulkan/vulkan.h>

#include <vector>

namespace Descriptors {

    VkDescriptorSetLayout createDescriptorSetLayout(VkDevice logicalDevice);

    VkDescriptorPool createDescriptorPool(VkDevice logicalDevice, uint32_t size);


    void createDescriptorSets(
        VkDevice logicalDevice,
        uint32_t size,
        VkDescriptorSetLayout descriptorSetLayout,
        VkDescriptorPool descriptorPool,
        const std::vector<VkBuffer>& uniformBuffers,
        const std::vector<VkBuffer>& agentBuffers,
        size_t agentBuffersSize,
        VkImageView textureImageView,
        VkSampler textureSampler,
        std::vector<VkDescriptorSet>& descriptorSets);

}

#endif
