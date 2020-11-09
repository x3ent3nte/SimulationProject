#ifndef INSERTION_SORT_UTIL_H
#define INSERTION_SORT_UTIL_H

#include <vulkan/vulkan.h>

#include <vector>

#define X_DIM 512

namespace InsertionSortUtil {

    struct ValueAndIndex {
        float value;
        uint32_t index;

        bool operator<(const ValueAndIndex& other) const;
    };

    struct Info {
        uint32_t offset;
        uint32_t dataSize;
    };

    VkDescriptorSetLayout createDescriptorSetLayout(VkDevice logicalDevice);

    VkDescriptorPool createDescriptorPool(VkDevice logicalDevice, size_t maxSet);

    VkDescriptorSet createDescriptorSet(
        VkDevice logicalDevice,
        VkDescriptorSetLayout& descriptorSetLayout,
        VkDescriptorPool& descriptorPool,
        VkBuffer valueAndIndexBuffer,
        VkBuffer wasSwappedBuffer,
        VkBuffer infoBuffer,
        size_t numberOfElements);

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
        size_t numberOfElements);
}

#endif
