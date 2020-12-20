#ifndef REDUCER_UTIL_H
#define REDUCER_UTIL_H

#include <vulkan/vulkan.h>

namespace ReducerUtil {

    constexpr size_t xDim = 256;

    VkDescriptorSetLayout createDescriptorSetLayout(VkDevice logicalDevice);

    VkDescriptorPool createDescriptorPool(VkDevice logicalDevice, size_t maxSets);

    VkDescriptorSet createDescriptorSet(
        VkDevice logicalDevice,
        VkDescriptorSetLayout descriptorSetLayout,
        VkDescriptorPool descriptorPool,
        VkBuffer inBuffer,
        VkBuffer outBuffer,
        VkBuffer numberOfElementsBuffer,
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
        VkBuffer numberOfElementsBuffer,
        VkBuffer numberOfElementsBufferHostVisible,
        uint32_t numberOfElements);
}

#endif
