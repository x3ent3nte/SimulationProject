#include <Simulator/ReduceUtil.h>

#include <Utils/Compute.h>

#include <stdexcept>

namespace {
    constexpr size_t numberOfBindings = 3;
} // namespace anonymous

VkDescriptorSetLayout ReduceUtil::createDescriptorSetLayout(VkDevice logicalDevice) {
    return Compute::createDescriptorSetLayout(logicalDevice, numberOfBindings);
}

VkDescriptorPool ReduceUtil::createDescriptorPool(VkDevice logicalDevice, size_t maxSets) {
    return Compute::createDescriptorPool(logicalDevice, numberOfBindings, maxSets);
}

VkDescriptorSet ReduceUtil::createDescriptoSet(
    VkDevice logicalDevice,
    VkDescriptorSetLayout descriptorSetLayout,
    VkDescriptorPool descriptorPool,
    VkBuffer inBuffer,
    VkBuffer outBuffer,
    VkBuffer dataSizeBuffer,
    size_t numberOfElements) {

    std::vector<Compute::BufferAndSize> bufferAndSizes = {
        {inBuffer, numberOfElements * sizeof(ReduceUtil::Collision)},
        {outBuffer, numberOfElements * sizeof(ReduceUtil::Collision)},
        {dataSizeBuffer, sizeof(uint32_t)}
    };

    return Compute::createDescriptorSet(
        logicalDevice,
        descriptorSetLayout,
        descriptorPool,
        bufferAndSizes);
}

VkPipeline ReduceUtil::createPipeline(
    VkDevice logicalDevice,
    VkPipelineLayout pipelineLayout) {

    return Compute::createPipeline("src/GLSL/Reduce.spv", logicalDevice, pipelineLayout);
}

VkCommandBuffer ReduceUtil::createCommandBuffer(
    VkDevice logicalDevice,
    VkCommandPool commandPool) {

    VkCommandBuffer commandBuffer;

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(logicalDevice, &commandBufferAllocateInfo, &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute command buffer");
    }

    return commandBuffer;
}
