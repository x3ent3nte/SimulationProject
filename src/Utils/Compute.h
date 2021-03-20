#ifndef COMPUTE_H
#define COMPUTE_H

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

namespace Compute {

    struct BufferAndSize {
        VkBuffer buffer;
        size_t size;
    };

    VkDescriptorSetLayout createDescriptorSetLayout(VkDevice logicalDevice, size_t numberOfBuffers);

    VkDescriptorPool createDescriptorPool(VkDevice logicalDevice, size_t numberOfBindings, size_t maxSets);

    VkDescriptorSet createDescriptorSet(
        VkDevice logicalDevice,
        VkDescriptorSetLayout& descriptorSetLayout,
        VkDescriptorPool& descriptorPool,
        const std::vector<Compute::BufferAndSize>& bufferAndSizes);

    VkPipelineLayout createPipelineLayout(VkDevice logicalDevice, VkDescriptorSetLayout descriptorSetLayout);

    VkPipelineLayout createPipelineLayoutWithPushConstant(
        VkDevice logicalDevice,
        VkDescriptorSetLayout descriptorSetLayout,
        uint32_t size);

    VkPipeline createPipeline(
        const std::string& shaderPath,
        VkDevice logicalDevice,
        VkPipelineLayout pipelineLayout);
}

#endif
