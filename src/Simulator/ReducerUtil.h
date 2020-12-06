#ifndef REDUCE_UTIL_H
#define REDUCE_UTIL_H

#include <vulkan/vulkan.h>

namespace ReducerUtil {

    constexpr size_t xDim = 256;

    struct Collision {
        uint32_t one;
        uint32_t two;
        float time;
    };

    VkDescriptorSetLayout createDescriptorSetLayout(VkDevice logicalDevice);

    VkDescriptorPool createDescriptorPool(VkDevice logicalDevice, size_t maxSets);

    VkDescriptorSet createDescriptorSet(
        VkDevice logicalDevice,
        VkDescriptorSetLayout descriptorSetLayout,
        VkDescriptorPool descriptorPool,
        VkBuffer inBuffer,
        VkBuffer outBuffer,
        VkBuffer dataSizeBuffer,
        uint32_t numberOfElements);

    VkPipeline createPipeline(
        VkDevice logicalDevice,
        VkPipelineLayout pipelineLayout);

    VkCommandBuffer createCommandBuffer(
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkPipeline pipeline,
        VkPipelineLayout pipelineLayout,
        VkDescriptorSet descriptorSet,
        VkBuffer dataSizeBuffer,
        VkBuffer dataSizeBufferHostVisible,
        uint32_t numberOfElements);
}

#endif
