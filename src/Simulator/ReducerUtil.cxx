#include <Simulator/ReducerUtil.h>

#include <Simulator/Collision.h>
#include <Utils/Compute.h>

#include <iostream>
#include <stdexcept>

namespace {
    constexpr size_t numberOfBindings = 3;
} // namespace anonymous

VkDescriptorSetLayout ReducerUtil::createDescriptorSetLayout(VkDevice logicalDevice) {
    return Compute::createDescriptorSetLayout(logicalDevice, numberOfBindings);
}

VkDescriptorPool ReducerUtil::createDescriptorPool(VkDevice logicalDevice, size_t maxSets) {
    return Compute::createDescriptorPool(logicalDevice, numberOfBindings, maxSets);
}

VkDescriptorSet ReducerUtil::createDescriptorSet(
    VkDevice logicalDevice,
    VkDescriptorSetLayout descriptorSetLayout,
    VkDescriptorPool descriptorPool,
    VkBuffer inBuffer,
    VkBuffer outBuffer,
    VkBuffer numberOfElementsBuffer,
    uint32_t numberOfElements) {

    std::vector<Compute::BufferAndSize> bufferAndSizes = {
        {inBuffer, numberOfElements * sizeof(Collision)},
        {outBuffer, numberOfElements * sizeof(Collision)},
        {numberOfElementsBuffer, sizeof(uint32_t)}
    };

    return Compute::createDescriptorSet(
        logicalDevice,
        descriptorSetLayout,
        descriptorPool,
        bufferAndSizes);
}

VkPipeline ReducerUtil::createPipeline(
    VkDevice logicalDevice,
    VkPipelineLayout pipelineLayout) {

    return Compute::createPipeline("src/GLSL/Reduce.spv", logicalDevice, pipelineLayout);
}

VkCommandBuffer ReducerUtil::createCommandBuffer(
    VkDevice logicalDevice,
    VkCommandPool commandPool,
    VkPipeline pipeline,
    VkPipelineLayout pipelineLayout,
    VkDescriptorSet descriptorSet,
    VkBuffer numberOfElementsBuffer,
    VkBuffer numberOfElementsBufferHostVisible,
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
    copyRegion.size = sizeof(uint32_t);
    vkCmdCopyBuffer(commandBuffer, numberOfElementsBufferHostVisible, numberOfElementsBuffer, 1, &copyRegion);

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

    uint32_t xGroups = ceil(((float) numberOfElements) / ((float) 2 * ReducerUtil::xDim));
    std::cout << "Number of Reduce X groups = " << xGroups << "\n";

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(commandBuffer, xGroups, 1, 1);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end compute command buffer");
    }

    return commandBuffer;
}
