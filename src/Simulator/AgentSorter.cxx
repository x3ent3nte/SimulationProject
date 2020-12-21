#include <Simulator/AgentSorter.h>

#include <Simulator/MapAgentToXUtil.h>
#include <Simulator/MapXToAgentUtil.h>
#include <Simulator/Agent.h>
#include <Utils/Buffer.h>
#include <Utils/Compute.h>
#include <Utils/Timer.h>

#include <array>
#include <stdexcept>

AgentSorter::AgentSorter(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    VkBuffer agentsBuffer,
    uint32_t numberOfElements)
    : m_logicalDevice(logicalDevice)
    , m_queue(queue)
    , m_commandPool(commandPool)
    , m_agentsBuffer(agentsBuffer)
    , m_insertionSorter(std::make_shared<InsertionSorter>(
        physicalDevice,
        logicalDevice,
        queue,
        commandPool,
        numberOfElements)) {

    m_currentNumberOfElements = numberOfElements;

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

    m_mapAgentToXDescriptorSetLayout = MapAgentToXUtil::createDescriptorSetLayout(m_logicalDevice);
    m_mapAgentToXDescriptorPool = MapAgentToXUtil::createDescriptorPool(m_logicalDevice, 1);
    m_mapAgentToXPipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_mapAgentToXDescriptorSetLayout);
    m_mapAgentToXPipeline = MapAgentToXUtil::createPipeline(m_logicalDevice, m_mapAgentToXPipelineLayout);
    m_mapAgentToXDescriptorSet = MapAgentToXUtil::createDescriptorSet(
        m_logicalDevice,
        m_mapAgentToXDescriptorSetLayout,
        m_mapAgentToXDescriptorPool,
        agentsBuffer,
        m_insertionSorter->m_valueAndIndexBuffer,
        m_timeDeltaBuffer,
        m_numberOfElementsBuffer,
        numberOfElements);

    m_mapXToAgentDescriptorSetLayout = MapXToAgentUtil::createDescriptorSetLayout(m_logicalDevice);
    m_mapXToAgentDescriptorPool = MapXToAgentUtil::createDescriptorPool(m_logicalDevice, 1);
    m_mapXToAgentPipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_mapXToAgentDescriptorSetLayout);
    m_mapXToAgentPipeline = MapXToAgentUtil::createPipeline(m_logicalDevice, m_mapXToAgentPipelineLayout);
    m_mapXToAgentDescriptorSet = MapXToAgentUtil::createDescriptorSet(
        m_logicalDevice,
        m_mapXToAgentDescriptorSetLayout,
        m_mapXToAgentDescriptorPool,
        m_insertionSorter->m_valueAndIndexBuffer,
        m_agentsBuffer,
        m_otherAgentsBuffer,
        m_numberOfElementsBuffer,
        numberOfElements);

    m_mapAgentToXCommandBuffer = VK_NULL_HANDLE;
    m_mapXToAgentCommandBuffer = VK_NULL_HANDLE;
    createCommandBuffers(numberOfElements);

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

void AgentSorter::createCommandBuffers(uint32_t numberOfElements) {
    std::array<VkCommandBuffer, 2> commandBuffers = {
        m_mapAgentToXCommandBuffer,
        m_mapXToAgentCommandBuffer};
    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, commandBuffers.size(), commandBuffers.data());

    m_mapAgentToXCommandBuffer = MapAgentToXUtil::createCommandBuffer(
        m_logicalDevice,
        m_commandPool,
        m_mapAgentToXPipeline,
        m_mapAgentToXPipelineLayout,
        m_mapAgentToXDescriptorSet,
        m_timeDeltaBuffer,
        m_timeDeltaBufferHostVisible,
        numberOfElements);

    m_mapXToAgentCommandBuffer = MapXToAgentUtil::createCommandBuffer(
        m_logicalDevice,
        m_commandPool,
        m_mapXToAgentPipeline,
        m_mapXToAgentPipelineLayout,
        m_mapXToAgentDescriptorSet,
        m_otherAgentsBuffer,
        m_agentsBuffer,
        numberOfElements);
}

void AgentSorter::updateNumberOfElementsIfNecessary(uint32_t numberOfElements) {
    if (m_currentNumberOfElements == numberOfElements) {
        return;
    }

    createCommandBuffers(numberOfElements);

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

void AgentSorter::mapAgentToX(float timeDelta) {
    void* dataMap;
    vkMapMemory(m_logicalDevice, m_timeDeltaDeviceMemoryHostVisible, 0, sizeof(float), 0, &dataMap);
    float timeDeltaCopy = timeDelta;
    memcpy(dataMap, &timeDeltaCopy, sizeof(float));
    vkUnmapMemory(m_logicalDevice, m_timeDeltaDeviceMemoryHostVisible);

    VkSubmitInfo submitInfoOne{};
    submitInfoOne.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfoOne.commandBufferCount = 1;
    submitInfoOne.pCommandBuffers = &m_mapAgentToXCommandBuffer;

    vkResetFences(m_logicalDevice, 1, &m_fence);

    if (vkQueueSubmit(m_queue, 1, &submitInfoOne, m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit mapAgentToX command buffer");
    }
    vkWaitForFences(m_logicalDevice, 1, &m_fence, VK_TRUE, UINT64_MAX);
}

void AgentSorter::mapXToAgent() {
    VkSubmitInfo submitInfoOne{};
    submitInfoOne.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfoOne.commandBufferCount = 1;
    submitInfoOne.pCommandBuffers = &m_mapXToAgentCommandBuffer;

    vkResetFences(m_logicalDevice, 1, &m_fence);

    if (vkQueueSubmit(m_queue, 1, &submitInfoOne, m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit mapXToAgent command buffer");
    }
    vkWaitForFences(m_logicalDevice, 1, &m_fence, VK_TRUE, UINT64_MAX);
}

AgentSorter::~AgentSorter() {
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

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_mapAgentToXDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_logicalDevice, m_mapAgentToXDescriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_mapAgentToXPipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice,  m_mapAgentToXPipeline, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_mapXToAgentDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_logicalDevice, m_mapXToAgentDescriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_mapXToAgentPipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice,  m_mapXToAgentPipeline, nullptr);

    std::array<VkCommandBuffer, 3> commandBuffers = {
        m_mapAgentToXCommandBuffer,
        m_mapXToAgentCommandBuffer,
        m_setNumberOfElementsCommandBuffer};
    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, commandBuffers.size(), commandBuffers.data());

    vkDestroyFence(m_logicalDevice, m_fence, nullptr);
}

void AgentSorter::run(float timeDelta, uint32_t numberOfElements) {
    //Timer timer("AgentSorter::run");
    updateNumberOfElementsIfNecessary(numberOfElements);
    mapAgentToX(timeDelta);
    m_insertionSorter->run(numberOfElements);
    mapXToAgent();
}
