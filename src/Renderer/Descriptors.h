#ifndef DESCRIPTORS_H
#define DESCRIPTORS_H

#include <Renderer/Texture.h>

#include <vulkan/vulkan.h>

#include <memory>
#include <vector>

namespace Descriptors {

    VkDescriptorSetLayout createDescriptorSetLayout(VkDevice logicalDevice, size_t numberOfTextureSamplers);

    VkDescriptorPool createDescriptorPool(VkDevice logicalDevice, size_t maxSets, size_t numberOfTextureSamplers);


    void createDescriptorSets(
        VkDevice logicalDevice,
        uint32_t size,
        VkDescriptorSetLayout descriptorSetLayout,
        VkDescriptorPool descriptorPool,
        const std::vector<VkBuffer>& uniformBuffers,
        const std::vector<VkBuffer>& agentBuffers,
        size_t agentBuffersSize,
        const std::vector<std::shared_ptr<Texture>>& textures,
        std::vector<VkDescriptorSet>& descriptorSets);

}

#endif
