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

uint32_t numberOfGroups(uint32_t current) {
    return ceil(((float) current) / ((float) xDim));
}

VkDescriptorSetLayout createDescriptorSetLayout(VkDevice logicalDevice) {
    return Compute::createDescriptorSetLayout(logicalDevice, kNumberOfBindings);
}

VkDescriptorPool createDescriptorPool(VkDevice logicalDevice, size_t maxSets) {
    return Compute::createDescriptorPool(logicalDevice, kNumberOfBindings, maxSets);
}

template <typename T>
VkDescriptorSet createDescriptorSet(
    VkDevice logicalDevice,
    VkDescriptorSetLayout descriptorSetLayout,
    VkDescriptorPool descriptorPool,
    VkBuffer dataBuffer,
    uint32_t numberOfElements) {

    std::vector<Compute::BufferAndSize> bufferAndSizes = {
        {dataBuffer, numberOfElements * sizeof(T)}
    };

    return Compute::createDescriptorSet(
        logicalDevice,
        descriptorSetLayout,
        descriptorPool,
        bufferAndSizes);
}

void createScanCommandRecursive(
    VkCommandBuffer commandBuffer,
    ScannerUtil::Info info,
    VkDescriptorSet descriptorSet,
    VkPipelineLayout pipelineLayout,
    VkPipeline scanPipeline,
    VkPipeline addOffsetsPipeline) {

    const uint32_t xGroups = numberOfGroups(info.numberOfElements);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scanPipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Info), &info);
    vkCmdDispatch(commandBuffer, xGroups, 1, 1);

    if (xGroups > 1) {

        VkMemoryBarrier memoryBarrier = {};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.pNext = nullptr;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

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

        Info recursiveInfo = {info.offsetOffset, info.offsetOffset + xGroups, xGroups};
        createScanCommandRecursive(commandBuffer, recursiveInfo, descriptorSet, pipelineLayout, scanPipeline, addOffsetsPipeline);

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

        ScannerUtil::Info addOffsetsInfo = {info.dataOffset + ScannerUtil::xDim, info.offsetOffset, info.numberOfElements - ScannerUtil::xDim};

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, addOffsetsPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Info), &addOffsetsInfo);
        vkCmdDispatch(commandBuffer, xGroups - 1, 1, 1);
    }
}

template <typename T>
std::string scanShaderPath() {
    throw std::runtime_error("No Scan shader file for that type");
}

template <typename T>
std::string scanAddOffsetsShaderPath() {
    throw std::runtime_error("No Scan Add Offsets shader file for that type");
}

template<>
std::string scanShaderPath<int32_t>() {
    return "src/GLSL/spv/ScanInt.spv";
}

template<>
std::string scanAddOffsetsShaderPath<int32_t>() {
    return "src/GLSL/spv/ScanIntAddOffsets.spv";
}

template <>
std::string scanShaderPath<glm::uvec4>() {
    return "src/GLSL/spv/ScanVec4.spv";
}

template<>
std::string scanAddOffsetsShaderPath<glm::uvec4>() {
    return "src/GLSL/spv/ScanVec4AddOffsets.spv";
}

} // namespace ScannerUtil

template <typename T>
Scanner<T>::Scanner(
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
        bufferNumberOfElements * sizeof(T),
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
    m_pipeline = Compute::createPipeline(ScannerUtil::scanShaderPath<T>(), m_logicalDevice, m_pipelineLayout);
    m_addOffsetsPipeline = Compute::createPipeline(ScannerUtil::scanAddOffsetsShaderPath<T>(), m_logicalDevice, m_pipelineLayout);

    m_descriptorSet = ScannerUtil::createDescriptorSet<T>(
        m_logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        m_dataBuffer,
        bufferNumberOfElements);

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    m_currentNumberOfElements = numberOfElements;
    createScanCommand(m_currentNumberOfElements);

    if (vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute fence");
    }
}

template <typename T>
Scanner<T>::~Scanner() {
    vkFreeMemory(m_logicalDevice, m_dataDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_dataBuffer, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_descriptorSetLayout, nullptr);

    vkDestroyDescriptorPool(m_logicalDevice, m_descriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_pipeline, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_addOffsetsPipeline, nullptr);

    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &m_commandBuffer);

    vkDestroyFence(m_logicalDevice, m_fence, nullptr);
}

template <typename T>
void Scanner<T>::createScanCommand(uint32_t numberOfElements) {
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = m_commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(m_logicalDevice, &commandBufferAllocateInfo, &m_commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute command buffer");
    }

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    if (vkBeginCommandBuffer(m_commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin compute command buffer");
    }

    ScannerUtil::Info info = {0, numberOfElements, numberOfElements};
    ScannerUtil::createScanCommandRecursive(m_commandBuffer, info, m_descriptorSet, m_pipelineLayout, m_pipeline, m_addOffsetsPipeline);

    if (vkEndCommandBuffer(m_commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end compute command buffer");
    }
}

template <typename T>
void Scanner<T>::createScanCommandIfNecessary(uint32_t numberOfElements) {
    if (m_currentNumberOfElements != numberOfElements) {
        vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &m_commandBuffer);
        createScanCommand(numberOfElements);
        m_currentNumberOfElements = numberOfElements;
    }
}

template <typename T>
void Scanner<T>::runScanCommand() {

    VkSubmitInfo submitInfoOne{};
    submitInfoOne.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfoOne.commandBufferCount = 1;
    submitInfoOne.pCommandBuffers = &m_commandBuffer;

    vkResetFences(m_logicalDevice, 1, &m_fence);

    if (vkQueueSubmit(m_queue, 1, &submitInfoOne, m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit insertion sort command buffer");
    }
    vkWaitForFences(m_logicalDevice, 1, &m_fence, VK_TRUE, UINT64_MAX);
}

template <typename T>
void Scanner<T>::run(uint32_t numberOfElements) {
    if (numberOfElements > 1) {
        createScanCommandIfNecessary(numberOfElements);
        runScanCommand();
    }
}

template <typename T>
void Scanner<T>::recordCommand(VkCommandBuffer commandBuffer, uint32_t numberOfElements) {
    ScannerUtil::Info info = {0, numberOfElements, numberOfElements};
    ScannerUtil::createScanCommandRecursive(commandBuffer, info, m_descriptorSet, m_pipelineLayout, m_pipeline, m_addOffsetsPipeline);
}

template class Scanner<int32_t>;
template class Scanner<glm::uvec4>;
