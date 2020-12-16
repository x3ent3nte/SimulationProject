#include <Simulator/MapXToAgentUtil.h>

#include <Simulator/Agent.h>
#include <Simulator/InsertionSorterUtil.h>

#include <Utils/Compute.h>

#include <stdexcept>

namespace {
    constexpr size_t kNumberOfBindings = 4;
} // end namespace anonymous

VkDescriptorSetLayout MapXToAgentUtil::createDescriptorSetLayout(VkDevice logicalDevice) {
    return Compute::createDescriptorSetLayout(logicalDevice, kNumberOfBindings);
}

VkDescriptorPool MapXToAgentUtil::createDescriptorPool(VkDevice logicalDevice, size_t maxSets) {
    return Compute::createDescriptorPool(logicalDevice, kNumberOfBindings, maxSets);
}

VkDescriptorSet MapXToAgentUtil::createDescriptorSet(
    VkDevice logicalDevice,
    VkDescriptorSetLayout descriptorSetLayout,
    VkDescriptorPool descriptorPool,
    VkBuffer valueAndIndexBuffer,
    VkBuffer agentsBufferIn,
    VkBuffer agentsBufferOut,
    VkBuffer numberOfElementsBuffer,
    uint32_t numberOfElements) {

    std::vector<Compute::BufferAndSize> bufferAndSizes = {
        {valueAndIndexBuffer, numberOfElements * sizeof(InsertionSorterUtil::ValueAndIndex)},
        {agentsBufferIn, numberOfElements * sizeof(Agent)},
        {agentsBufferOut, numberOfElements * sizeof(Agent)},
        {numberOfElementsBuffer, sizeof(uint32_t)}
    };

    return Compute::createDescriptorSet(
        logicalDevice,
        descriptorSetLayout,
        descriptorPool,
        bufferAndSizes);
}

VkPipeline MapXToAgentUtil::createPipeline(
    VkDevice logicalDevice,
    VkPipelineLayout pipelineLayout) {

    return Compute::createPipeline("src/GLSL/MapXToAgent.spv", logicalDevice, pipelineLayout);
}

VkCommandBuffer MapXToAgentUtil::createCommandBuffer(
    VkDevice logicalDevice,
    VkCommandPool commandPool,
    VkPipeline pipeline,
    VkPipelineLayout pipelineLayout,
    VkDescriptorSet descriptorSet,
    uint32_t numberOfElements) {

    VkCommandBuffer commandBuffer;

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(logicalDevice, &commandBufferAllocateInfo, &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute command buffer");
    }

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin compute command buffer");
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    uint32_t xGroups = ceil(((float) numberOfElements) / ((float) MapXToAgentUtil::xDim));

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(commandBuffer, xGroups, 1, 1);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end compute command buffer");
    }

    return commandBuffer;
}
