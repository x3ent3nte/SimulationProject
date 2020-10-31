#ifndef INSERTION_SORT_H
#define INSERTION_SORT_H

#include <vulkan/vulkan.h>

namespace InsertionSort {

    struct ValueAndIndex {
        float value;
        uint32_t index;
    };

    struct Info {
        uint32_t wasSwapped;
        uint32_t dataSize;
    };

    VkDescriptorSetLayout createDescriptorSetLayout(VkDevice logicalDevice);

    VkDescriptorPool createDescriptorPool(VkDevice logicalDevice, size_t maxSets);

    VkDescriptorSet createDescriptorSet(
        VkDevice logicalDevice,
        VkDescriptorSetLayout& descriptorSetLayout,
        VkDescriptorPool& descriptorPool,
        VkBuffer valueAndIndexBuffer,
        VkBuffer infoBuffer,
        size_t numberOfElements);

    VkPipelineLayout createPipelineLayout(VkDevice logicalDevice, VkDescriptorSetLayout descriptorSetLayout);

    VkPipeline createPipeline(
        VkDevice logicalDevice,
        VkShaderModule shaderModule,
        VkDescriptorSetLayout descriptorSetLayout,
        VkPipelineLayout pipelineLayout);
}

#endif
