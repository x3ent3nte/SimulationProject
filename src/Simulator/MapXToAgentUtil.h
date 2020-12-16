#ifndef MAP_X_TO_AGENT_UTIL_H
#define MAP_X_TO_AGENT_UTIL_H

#include <vulkan/vulkan.h>

namespace MapXToAgentUtil {

    constexpr size_t xDim = 256;

    VkDescriptorSetLayout createDescriptorSetLayout(VkDevice logicalDevice);

    VkDescriptorPool createDescriptorPool(VkDevice logicalDevice, size_t maxSets);

    VkDescriptorSet createDescriptorSet(
        VkDevice logicalDevice,
        VkDescriptorSetLayout descriptorSetLayout,
        VkDescriptorPool descriptorPool,
        VkBuffer valueAndIndexBuffer,
        VkBuffer agentsBufferIn,
        VkBuffer agentsBufferOut,
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
        uint32_t numberOfElements);
}

#endif
