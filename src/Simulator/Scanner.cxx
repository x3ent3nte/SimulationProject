#include <Simulator/Scanner.h>

#include <Utils/Buffer.h>
#include <Utils/Compute.h>

#include <vector>
#include <stdexcept>

namespace ScannerUtil {

constexpr size_t xDim = 512;
constexpr size_t kNumberOfBindings = 1;

struct Info {
    uint32_t dataOffset;
    uint32_t offsetOffset;
    uint32_t numberOfElements;
};

VkDescriptorSetLayout createDescriptorSetLayout(VkDevice logicalDevice) {
    return Compute::createDescriptorSetLayout(logicalDevice, kNumberOfBindings);
}

VkDescriptorPool createDescriptorPool(VkDevice logicalDevice, size_t maxSets) {
    return Compute::createDescriptorPool(logicalDevice, kNumberOfBindings, maxSets);
}

VkDescriptorSet createDescriptorSet(
    VkDevice logicalDevice,
    VkDescriptorSetLayout descriptorSetLayout,
    VkDescriptorPool descriptorPool,
    VkBuffer dataBuffer,
    uint32_t numberOfElements) {

    std::vector<Compute::BufferAndSize> bufferAndSizes = {
        {dataBuffer, numberOfElements * sizeof(int32_t)}
    };

    return Compute::createDescriptorSet(
        logicalDevice,
        descriptorSetLayout,
        descriptorPool,
        bufferAndSizes);
}

VkCommandBuffer createCommandBuffer(
    VkDevice logicalDevice,
    VkCommandPool commandPool,
    VkPipeline pipeline,
    VkPipelineLayout pipelineLayout,
    VkDescriptorSet descriptorSet,
    VkBuffer dataBuffer,
    Info info) {

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

    /*
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = sizeof(ScannerUtil::Info);
    vkCmdCopyBuffer(commandBuffer, infoBufferHostVisible, infoBuffer, 1, &copyRegion);

    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.pNext = nullptr;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &memoryBarrier,
        0,
        nullptr,
        0,
        nullptr);
    */
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    uint32_t xGroups = ceil(((float) info.numberOfElements) / ((float) ScannerUtil::xDim));

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Info), &info);
    vkCmdDispatch(commandBuffer, xGroups, 1, 1);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end compute command buffer");
    }

    return commandBuffer;
}

} // namespace ScannerUtil

Scanner::Scanner(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t numberOfElements) {

    m_logicalDevice = logicalDevice;
    m_queue = queue;
    m_commandPool = commandPool;

    size_t bufferNumberOfElements = numberOfElements * 2;

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        bufferNumberOfElements * sizeof(int32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_dataBuffer,
        m_dataDeviceMemory);

    m_descriptorSetLayout = ScannerUtil::createDescriptorSetLayout(m_logicalDevice);
    m_descriptorPool = ScannerUtil::createDescriptorPool(m_logicalDevice, 1);
    m_pipelineLayout = Compute::createPipelineLayoutWithPushConstant(
        m_logicalDevice,
        m_descriptorSetLayout,
        sizeof(ScannerUtil::Info));
    m_pipeline = Compute::createPipeline("src/GLSL/spv/Scan.spv", m_logicalDevice, m_pipelineLayout);
    m_addOffsetsPipeline = Compute::createPipeline("src/GLSL/spv/ScanAddOffsets.spv", m_logicalDevice, m_pipelineLayout);

    m_descriptorSet = ScannerUtil::createDescriptorSet(
        m_logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        m_dataBuffer,
        bufferNumberOfElements);

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute fence");
    }
}

Scanner::~Scanner() {
    vkFreeMemory(m_logicalDevice, m_dataDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_dataBuffer, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_descriptorSetLayout, nullptr);

    vkDestroyDescriptorPool(m_logicalDevice, m_descriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_pipeline, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_addOffsetsPipeline, nullptr);

    vkDestroyFence(m_logicalDevice, m_fence, nullptr);
}

void Scanner::addOffsets(uint32_t dataOffset, uint32_t offsetOffset, uint32_t numberOfElements) {

    VkCommandBuffer commandBuffer = ScannerUtil::createCommandBuffer(
        m_logicalDevice,
        m_commandPool,
        m_addOffsetsPipeline,
        m_pipelineLayout,
        m_descriptorSet,
        m_dataBuffer,
        {dataOffset, offsetOffset, numberOfElements});

    VkSubmitInfo submitInfoOne{};
    submitInfoOne.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfoOne.commandBufferCount = 1;
    submitInfoOne.pCommandBuffers = &commandBuffer;

    vkResetFences(m_logicalDevice, 1, &m_fence);

    if (vkQueueSubmit(m_queue, 1, &submitInfoOne, m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit insertion sort command buffer");
    }
    vkWaitForFences(m_logicalDevice, 1, &m_fence, VK_TRUE, UINT64_MAX);

    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &commandBuffer);
}

void Scanner::runScanCommand(uint32_t dataOffset, uint32_t offsetOffset, uint32_t numberOfElements) {

    VkCommandBuffer commandBuffer = ScannerUtil::createCommandBuffer(
        m_logicalDevice,
        m_commandPool,
        m_pipeline,
        m_pipelineLayout,
        m_descriptorSet,
        m_dataBuffer,
        {dataOffset, offsetOffset, numberOfElements});

    VkSubmitInfo submitInfoOne{};
    submitInfoOne.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfoOne.commandBufferCount = 1;
    submitInfoOne.pCommandBuffers = &commandBuffer;

    vkResetFences(m_logicalDevice, 1, &m_fence);

    if (vkQueueSubmit(m_queue, 1, &submitInfoOne, m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit insertion sort command buffer");
    }
    vkWaitForFences(m_logicalDevice, 1, &m_fence, VK_TRUE, UINT64_MAX);

    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &commandBuffer);

    uint32_t xGroups = ceil(((float) numberOfElements) / ((float) ScannerUtil::xDim));

    if (xGroups > 1) {
        runScanCommand(dataOffset + numberOfElements, offsetOffset + xGroups, xGroups);
        addOffsets(dataOffset + ScannerUtil::xDim, offsetOffset, numberOfElements - ScannerUtil::xDim);
    }
}

void Scanner::run(uint32_t numberOfElements) {
    if (numberOfElements > 1) {
        runScanCommand(0, numberOfElements, numberOfElements);
    }
}
