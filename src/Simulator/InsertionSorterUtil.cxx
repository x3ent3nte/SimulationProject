#include <Simulator/InsertionSorterUtil.h>

#include <Simulator/Collision.h>
#include <Utils/Compute.h>

#include <array>
#include <stdexcept>
#include <iostream>

namespace {
    constexpr size_t numberOfBindings = 4;
} // namespace anonymous

VkDescriptorSetLayout InsertionSorterUtil::createDescriptorSetLayout(VkDevice logicalDevice) {
    return Compute::createDescriptorSetLayout(logicalDevice, numberOfBindings);
}

VkDescriptorPool InsertionSorterUtil::createDescriptorPool(VkDevice logicalDevice, size_t maxSets) {
    return Compute::createDescriptorPool(logicalDevice, numberOfBindings, maxSets);
}

VkDescriptorSet InsertionSorterUtil::createDescriptorSet(
    VkDevice logicalDevice,
    VkDescriptorSetLayout descriptorSetLayout,
    VkDescriptorPool descriptorPool,
    VkBuffer valueAndIndexBuffer,
    VkBuffer wasSwappedBuffer,
    VkBuffer numberOfElementsBuffer,
    VkBuffer offsetBuffer,
    uint32_t numberOfElements) {

    std::vector<Compute::BufferAndSize> bufferAndSizes = {
        {valueAndIndexBuffer, numberOfElements * sizeof(ValueAndIndex)},
        {wasSwappedBuffer, sizeof(uint32_t)},
        {numberOfElementsBuffer, sizeof(uint32_t)},
        {offsetBuffer, sizeof(uint32_t)}
    };

    return Compute::createDescriptorSet(
        logicalDevice,
        descriptorSetLayout,
        descriptorPool,
        bufferAndSizes);
}

VkPipeline InsertionSorterUtil::createPipeline(
    VkDevice logicalDevice,
    VkPipelineLayout pipelineLayout) {

    return Compute::createPipeline("src/GLSL/InsertionSort.spv", logicalDevice, pipelineLayout);
}

VkCommandBuffer InsertionSorterUtil::createCommandBuffer(
    VkDevice logicalDevice,
    VkCommandPool commandPool,
    VkPipeline pipeline,
    VkPipelineLayout pipelineLayout,
    VkDescriptorSet descriptorSetOne,
    VkDescriptorSet descriptorSetTwo,
    VkBuffer valueAndIndexBuffer,
    VkBuffer wasSwappedBuffer,
    VkBuffer wasSwappedBufferHostVisible,
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
    vkCmdCopyBuffer(commandBuffer, wasSwappedBufferHostVisible, wasSwappedBuffer, 1, &copyRegion);

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

    uint32_t xGroups = ceil(((float) numberOfElements) / ((float) 2 * InsertionSorterUtil::xDim));
    std::cout << "Number of X groups = " << xGroups << "\n";

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSetOne, 0, nullptr);
    vkCmdDispatch(commandBuffer, xGroups, 1, 1);

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

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSetTwo, 0, nullptr);
    vkCmdDispatch(commandBuffer, xGroups, 1, 1);

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

    vkCmdCopyBuffer(commandBuffer, wasSwappedBuffer, wasSwappedBufferHostVisible, 1, &copyRegion);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end compute command buffer");
    }

    return commandBuffer;
}
