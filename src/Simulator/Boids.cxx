#include <Simulator/Boids.h>

#include <Simulator/Agent.h>
#include <Utils/Buffer.h>
#include <Utils/Compute.h>
#include <Utils/Timer.h>

#include <array>
#include <stdexcept>
#include <iostream>

namespace BoidsUtil {
    size_t xDim = 512;
    size_t kNumberOfBindings = 6;

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
        VkBuffer agentsInBuffer,
        VkBuffer agentsOutBuffer,
        VkBuffer reproductionBuffer,
        VkBuffer timeDeltaBuffer,
        VkBuffer numberOfElementsBuffer,
        VkBuffer playerInputStatesBuffer,
        uint32_t numberOfElements,
        uint32_t maxNumberOfPlayers) {

        std::vector<Compute::BufferAndSize> bufferAndSizes = {
            {agentsInBuffer, numberOfElements * sizeof(Agent)},
            {agentsOutBuffer, numberOfElements * sizeof(Agent)},
            {reproductionBuffer, numberOfElements * sizeof(uint32_t)},
            {timeDeltaBuffer, sizeof(float)},
            {numberOfElementsBuffer, sizeof(uint32_t)},
            {playerInputStatesBuffer, maxNumberOfPlayers * sizeof(uint32_t)}
        };

        return Compute::createDescriptorSet(
            logicalDevice,
            descriptorSetLayout,
            descriptorPool,
            bufferAndSizes);
    }

    VkPipeline createPipeline(VkDevice logicalDevice, VkPipelineLayout pipelineLayout) {
        return Compute::createPipeline("src/GLSL/spv/Boids.spv", logicalDevice, pipelineLayout);
    }

    VkCommandBuffer createCommandBuffer(
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkPipeline pipeline,
        VkPipelineLayout pipelineLayout,
        VkDescriptorSet descriptorSet,
        VkBuffer agentsBuffer,
        VkBuffer otherAgentsBuffer,
        VkBuffer timeDeltaBuffer,
        VkBuffer timeDeltaHostVisibleBuffer,
        std::shared_ptr<Scanner> scanner,
        std::shared_ptr<Reproducer> reproducer,
        uint32_t numberOfElements) {

        VkCommandBuffer commandBuffer;

        VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.commandPool = commandPool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1;

        VkMemoryBarrier memoryBarrier = {};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.pNext = nullptr;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

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
            1,
            &memoryBarrier,
            0,
            nullptr,
            0,
            nullptr);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

        uint32_t xGroups = ceil(((float) numberOfElements) / ((float) xDim));

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        vkCmdDispatch(commandBuffer, xGroups, 1, 1);

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

        scanner->recordCommand(commandBuffer, numberOfElements);

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

        reproducer->recordCommand(commandBuffer, numberOfElements);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to end compute command buffer");
        }

        return commandBuffer;
    }

} // end namespace BoidsUtil

Boids::Boids(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    VkBuffer agentsBuffer,
    uint32_t numberOfElements,
    uint32_t maxNumberOfPlayers)
    : m_logicalDevice(logicalDevice)
    , m_queue(queue)
    , m_commandPool(commandPool)
    , m_agentsBuffer(agentsBuffer)
    , m_currentNumberOfElements(numberOfElements)
    , m_maxNumberOfPlayers(maxNumberOfPlayers) {

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        numberOfElements * sizeof(Agent),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_otherAgentsBuffer,
        m_otherAgentsDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_timeDeltaBuffer,
        m_timeDeltaDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_timeDeltaBufferHostVisible,
        m_timeDeltaDeviceMemoryHostVisible);

    Buffer::createBufferWithData(
        &numberOfElements,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        m_logicalDevice,
        commandPool,
        queue,
        m_numberOfElementsBuffer,
        m_numberOfElementsDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_numberOfElementsBufferHostVisible,
        m_numberOfElementsDeviceMemoryHostVisible);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        m_maxNumberOfPlayers * sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_playerInputStatesBuffer,
        m_playerInputStatesDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        m_maxNumberOfPlayers * sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_playerInputStatesHostVisibleBuffer,
        m_playerInputStatesHostVisibleDeviceMemory);

    m_scanner = std::make_shared<Scanner>(
        physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        numberOfElements);

    m_reproducer = std::make_shared<Reproducer>(
        logicalDevice,
        m_otherAgentsBuffer,
        m_scanner->m_dataBuffer,
        agentsBuffer,
        numberOfElements);

    m_descriptorSetLayout = BoidsUtil::createDescriptorSetLayout(m_logicalDevice);
    m_descriptorPool = BoidsUtil::createDescriptorPool(m_logicalDevice, 1);
    m_pipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_descriptorSetLayout);
    m_pipeline = BoidsUtil::createPipeline(m_logicalDevice, m_pipelineLayout);
    m_descriptorSet = BoidsUtil::createDescriptorSet(
        m_logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        m_agentsBuffer,
        m_otherAgentsBuffer,
        m_scanner->m_dataBuffer,
        m_timeDeltaBuffer,
        m_numberOfElementsBuffer,
        m_playerInputStatesBuffer,
        numberOfElements,
        m_maxNumberOfPlayers);

    createCommandBuffer(numberOfElements);

    m_setNumberOfElementsCommandBuffer = Buffer::recordCopyCommand(
        m_logicalDevice,
        m_commandPool,
        m_numberOfElementsBufferHostVisible,
        m_numberOfElementsBuffer,
        sizeof(uint32_t));

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute fence");
    }
}

Boids::~Boids() {
    vkFreeMemory(m_logicalDevice, m_otherAgentsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_otherAgentsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_timeDeltaDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_timeDeltaBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_timeDeltaDeviceMemoryHostVisible, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_timeDeltaBufferHostVisible, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfElementsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfElementsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfElementsDeviceMemoryHostVisible, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfElementsBufferHostVisible, nullptr);

    vkFreeMemory(m_logicalDevice, m_playerInputStatesDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_playerInputStatesBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_playerInputStatesHostVisibleDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_playerInputStatesHostVisibleBuffer, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_logicalDevice, m_descriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice,  m_pipeline, nullptr);

    std::array<VkCommandBuffer, 2> commandBuffers = {
        m_commandBuffer,
        m_setNumberOfElementsCommandBuffer};
    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, commandBuffers.size(), commandBuffers.data());

    vkDestroyFence(m_logicalDevice, m_fence, nullptr);
}

void Boids::updateNumberOfElementsIfNecessary(uint32_t numberOfElements) {
    if (m_currentNumberOfElements == numberOfElements) {
        return;
    }

    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &m_commandBuffer);

    createCommandBuffer(numberOfElements);

    m_currentNumberOfElements = numberOfElements;

    void* dataMap;
    vkMapMemory(m_logicalDevice, m_numberOfElementsDeviceMemoryHostVisible, 0, sizeof(uint32_t), 0, &dataMap);
    uint32_t numberOfElementsCopy = numberOfElements;
    memcpy(dataMap, &numberOfElementsCopy, sizeof(uint32_t));
    vkUnmapMemory(m_logicalDevice, m_numberOfElementsDeviceMemoryHostVisible);

    VkSubmitInfo submitInfoOne{};
    submitInfoOne.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfoOne.commandBufferCount = 1;
    submitInfoOne.pCommandBuffers = &m_setNumberOfElementsCommandBuffer;

    vkResetFences(m_logicalDevice, 1, &m_fence);

    if (vkQueueSubmit(m_queue, 1, &submitInfoOne, m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit insertion sort set data size command buffer");
    }
    vkWaitForFences(m_logicalDevice, 1, &m_fence, VK_TRUE, UINT64_MAX);
}

void Boids::createCommandBuffer(uint32_t numberOfElements) {
    m_commandBuffer = BoidsUtil::createCommandBuffer(
        m_logicalDevice,
        m_commandPool,
        m_pipeline,
        m_pipelineLayout,
        m_descriptorSet,
        m_agentsBuffer,
        m_otherAgentsBuffer,
        m_timeDeltaBuffer,
        m_timeDeltaBufferHostVisible,
        m_scanner,
        m_reproducer,
        numberOfElements);
}

uint32_t Boids::extractNumberOfElements() {
    Timer timer("Boids::extractNumberOfElements");

    VkCommandBuffer copyCommand = Buffer::recordCopyCommand(
        m_logicalDevice,
        m_commandPool,
        m_scanner->m_dataBuffer,
        m_numberOfElementsBufferHostVisible,
        sizeof(uint32_t),
        (m_currentNumberOfElements - 1) * sizeof(uint32_t),
        0);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &copyCommand;
    vkResetFences(m_logicalDevice, 1, &m_fence);

    vkQueueSubmit(m_queue, 1, &submitInfo, m_fence);

    vkWaitForFences(m_logicalDevice, 1, &m_fence, VK_TRUE, UINT64_MAX);

    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &copyCommand);

    void* dataMap;
    vkMapMemory(m_logicalDevice, m_numberOfElementsDeviceMemoryHostVisible, 0, sizeof(uint32_t), 0, &dataMap);
    uint32_t numberOfElements;
    memcpy(&numberOfElements, dataMap, sizeof(uint32_t));
    vkUnmapMemory(m_logicalDevice, m_numberOfElementsDeviceMemoryHostVisible);
    return numberOfElements;
}

void Boids::copyPlayerInputStates(std::vector<uint32_t>& playerInputStates) {
    Timer timer("Boids::copyPlayerInputStates");
    const size_t numberOfPlayers = playerInputStates.size();
    const size_t memorySize = numberOfPlayers * sizeof(uint32_t);

    void* dataMap;
    vkMapMemory(m_logicalDevice, m_playerInputStatesHostVisibleDeviceMemory, 0, memorySize, 0, &dataMap);
    memcpy(dataMap, playerInputStates.data(), memorySize);
    vkUnmapMemory(m_logicalDevice, m_playerInputStatesHostVisibleDeviceMemory);

    VkCommandBuffer copyCommand = Buffer::recordCopyCommand(
        m_logicalDevice,
        m_commandPool,
        m_playerInputStatesHostVisibleBuffer,
        m_playerInputStatesBuffer,
        memorySize);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &copyCommand;
    vkResetFences(m_logicalDevice, 1, &m_fence);

    vkQueueSubmit(m_queue, 1, &submitInfo, m_fence);

    vkWaitForFences(m_logicalDevice, 1, &m_fence, VK_TRUE, UINT64_MAX);

    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &copyCommand);
}

uint32_t Boids::run(float timeDelta, uint32_t numberOfElements, std::vector<uint32_t>& playerInputStates) {
    updateNumberOfElementsIfNecessary(numberOfElements);
    copyPlayerInputStates(playerInputStates);

    void* dataMap;
    vkMapMemory(m_logicalDevice, m_timeDeltaDeviceMemoryHostVisible, 0, sizeof(float), 0, &dataMap);
    float timeDeltaCopy = timeDelta;
    memcpy(dataMap, &timeDeltaCopy, sizeof(float));
    vkUnmapMemory(m_logicalDevice, m_timeDeltaDeviceMemoryHostVisible);

    VkSubmitInfo submitInfoOne{};
    submitInfoOne.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfoOne.commandBufferCount = 1;
    submitInfoOne.pCommandBuffers = &m_commandBuffer;

    vkResetFences(m_logicalDevice, 1, &m_fence);

    if (vkQueueSubmit(m_queue, 1, &submitInfoOne, m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit mapAgentToX command buffer");
    }
    vkWaitForFences(m_logicalDevice, 1, &m_fence, VK_TRUE, UINT64_MAX);

    return extractNumberOfElements();
}
