#ifndef REDUCE_UTIL_H
#define REDUCE_UTIL_H

#include <vulkan/vulkan.h>

namespace ReduceUtil {

    struct Collision {
        uint32_t one;
        uint32_t two;
        float time;
    };

    VkDescriptorSetLayout createDescriptorSetLayout(VkDevice logicalDevice);

    VkDescriptorPool createDescriptorPool(VkDevice logicalDevice, size_t maxSets);

    VkDescriptorSet createDescriptoSet(
        VkDevice logicalDevice,
        VkDescriptorSetLayout descriptorSetLayout,
        VkDescriptorPool descriptorPool,
        VkBuffer inBuffer,
        VkBuffer outBuffer,
        VkBuffer dataSizeBuffer,
        size_t numberOfElements);

    VkPipeline createPipeline(
        VkDevice logicalDevice,
        VkPipelineLayout pipelineLayout);

    VkCommandBuffer createCommandBuffer(
        VkDevice logicalDevice,
        VkCommandPool commandPool);
}

#endif
