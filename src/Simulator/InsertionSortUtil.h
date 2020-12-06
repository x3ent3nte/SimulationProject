#ifndef INSERTION_SORT_UTIL_H
#define INSERTION_SORT_UTIL_H

#include <vulkan/vulkan.h>

#include <vector>

namespace InsertionSortUtil {

    constexpr size_t xDim = 256;

    struct ValueAndIndex {
        float value;
        uint32_t index;

        bool operator<(const ValueAndIndex& other) const;
    };

    VkDescriptorSetLayout createDescriptorSetLayout(VkDevice logicalDevice);

    VkDescriptorPool createDescriptorPool(VkDevice logicalDevice, size_t maxSets);

    VkDescriptorSet createDescriptorSet(
        VkDevice logicalDevice,
        VkDescriptorSetLayout descriptorSetLayout,
        VkDescriptorPool descriptorPool,
        VkBuffer valueAndIndexBuffer,
        VkBuffer wasSwappedBuffer,
        VkBuffer dataSizeBuffer,
        VkBuffer offsetBuffer,
        uint32_t numberOfElements);

    VkPipeline createPipeline(
        VkDevice logicalDevice,
        VkPipelineLayout pipelineLayout);

    VkCommandBuffer createCommandBuffer(
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkPipeline pipeline,
        VkPipelineLayout pipelineLayout,
        VkDescriptorSet descriptorSetOne,
        VkDescriptorSet descriptorSetTwo,
        VkBuffer valueAndIndexBuffer,
        VkBuffer wasSwappedBuffer,
        VkBuffer wasSwappedBufferHostVisible,
        uint32_t numberOfElements);
}

#endif
