#include <Simulator/MapAgentToXUtil.h>

#include <Simulator/Agent.h>
#include <Simulator/Collision.h>
#include <Simulator/InsertionSorterUtil.h>

#include <Utils/Compute.h>

#include <stdexcept>

namespace {
    constexpr size_t kNumberOfBindings = 4;
} // end namespace anonymous

VkDescriptorSetLayout MapAgentToXUtil::createDescriptorSetLayout(VkDevice logicalDevice) {
    return Compute::createDescriptorSetLayout(logicalDevice, kNumberOfBindings);
}

VkDescriptorPool MapAgentToXUtil::createDescriptorPool(VkDevice logicalDevice, size_t maxSets) {
    return Compute::createDescriptorPool(logicalDevice, kNumberOfBindings, maxSets);
}

VkDescriptorSet MapAgentToXUtil::createDescriptorSet(
    VkDevice logicalDevice,
    VkDescriptorSetLayout descriptorSetLayout,
    VkDescriptorPool descriptorPool,
    VkBuffer agentsBuffer,
    VkBuffer valueAndIndexBuffer,
    VkBuffer timeDeltaBuffer,
    VkBuffer numberOfElementsBuffer,
    uint32_t numberOfElements) {

    std::vector<Compute::BufferAndSize> bufferAndSizes = {
        {agentsBuffer, numberOfElements * sizeof(Agent)},
        {valueAndIndexBuffer, numberOfElements * sizeof(ValueAndIndex)},
        {timeDeltaBuffer, sizeof(float)},
        {numberOfElementsBuffer, sizeof(uint32_t)}
    };

    return Compute::createDescriptorSet(
        logicalDevice,
        descriptorSetLayout,
        descriptorPool,
        bufferAndSizes);
}

VkPipeline MapAgentToXUtil::createPipeline(
    VkDevice logicalDevice,
    VkPipelineLayout pipelineLayout) {

    return Compute::createPipeline("src/GLSL/MapAgentToX.spv", logicalDevice, pipelineLayout);
}

VkCommandBuffer MapAgentToXUtil::createCommandBuffer(
    VkDevice logicalDevice,
    VkCommandPool commandPool,
    VkPipeline pipeline,
    VkPipelineLayout pipelineLayout,
    VkDescriptorSet descriptorSet,
    VkBuffer timeDeltaBuffer,
    VkBuffer timeDeltaHostVisibleBuffer,
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

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = sizeof(float);
    vkCmdCopyBuffer(commandBuffer, timeDeltaHostVisibleBuffer, timeDeltaBuffer, 1, &copyRegion);

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        0,
        nullptr);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    uint32_t xGroups = ceil(((float) numberOfElements) / ((float) MapAgentToXUtil::xDim));

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(commandBuffer, xGroups, 1, 1);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end compute command buffer");
    }

    return commandBuffer;
}
