#ifndef COLLIDER_UTIL_H
#define COLLIDER_UTIL_H

#include <vulkan/vulkan.h>

namespace ColliderUtil {

    constexpr size_t xDim = 256;

    VkDescriptorSetLayout createDescriptorSetLayout(VkDevice logicalDevice);

    VkDescriptorPool createDescriptorPool(VkDevice logicalDevice, size_t maxSets);

    VkDescriptorSet createDescriptorSet(
        VkDevice logicalDevice,
        VkDescriptorSetLayout descriptorSetLayout,
        VkDescriptorPool descriptorPool,
        VkBuffer agentsBuffer,
        VkBuffer collisionsBuffer,
        VkBuffer timeDeltaBuffer,
        VkBuffer numberOfElementsBuffer,
        uint32_t numberOfElements);

    VkPipeline createPipeline(VkDevice logicalDevice, VkPipelineLayout pipelineLayout);

    VkCommandBuffer createCommandBuffer(
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkPipeline pipeline,
        VkPipelineLayout pipelineLayout,
        VkDescriptorSet descriptorSet,
        VkBuffer timeDeltaBuffer,
        VkBuffer timeDeltaHostVisibleBuffer,
        uint32_t numberOfElements);
}

#endif
