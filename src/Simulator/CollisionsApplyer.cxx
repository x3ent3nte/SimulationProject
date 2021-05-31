#include <Simulator/CollisionsApplyer.h>

#include <Simulator/ComputedCollision.h>
#include <Utils/Buffer.h>
#include <Utils/Command.h>
#include <Utils/Compute.h>

#include <array>

namespace CollisionsApplyerUtil {

    constexpr size_t xDim = 256;
    constexpr uint32_t kMaxCollisionsPerAgent = 10;

    constexpr size_t kRadixTimeMapNumberOfBindings = 3;
    constexpr size_t kRadixGatherNumberOfBindings = 4;

    constexpr size_t kNumberOfBindings = 5;

    VkCommandBuffer createRadixTimeMapCommandBuffer(
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkPipeline pipeline,
        VkPipelineLayout pipelineLayout,
        VkDescriptorSet descriptorSet,
        VkBuffer numberOfElementsHostVisibleBuffer,
        VkBuffer numberOfElementsBuffer,
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
        vkCmdCopyBuffer(commandBuffer, numberOfElementsHostVisibleBuffer, numberOfElementsBuffer, 1, &copyRegion);

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

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

        uint32_t xGroups = ceil(((float) numberOfElements) / ((float) xDim));

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        vkCmdDispatch(commandBuffer, xGroups, 1, 1);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to end compute command buffer");
        }

        return commandBuffer;
    }

    VkCommandBuffer createRadixGatherCommandBuffer(
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

        uint32_t xGroups = ceil(((float) numberOfElements) / ((float) xDim));

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        vkCmdDispatch(commandBuffer, xGroups, 1, 1);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to end compute command buffer");
        }

        return commandBuffer;
    }

}

CollisionsApplyer::CollisionsApplyer(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    VkBuffer agentsBuffer,
    uint32_t maxNumberOfAgents) {

    m_currentNumberOfAgents = maxNumberOfAgents;
    m_currentNumberOfCollisions = maxNumberOfAgents * CollisionsApplyerUtil::kMaxCollisionsPerAgent;

    m_logicalDevice = logicalDevice;
    m_queue = queue;
    m_commandPool = commandPool;

    m_radixSorter = std::make_shared<RadixSorter>(
        physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        maxNumberOfAgents * CollisionsApplyerUtil::kMaxCollisionsPerAgent);

    const size_t computedCollisionsMemorySize = maxNumberOfAgents * 2 * CollisionsApplyerUtil::kMaxCollisionsPerAgent * sizeof(ComputedCollision);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        computedCollisionsMemorySize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_computedCollisionsBuffer,
        m_computedCollisionsDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        computedCollisionsMemorySize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_otherComputedCollisionsBuffer,
        m_otherComputedCollisionsDeviceMemory);

    Buffer::createBufferWithData(
        &m_currentNumberOfCollisions,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        m_logicalDevice,
        m_commandPool,
        m_queue,
        m_numberOfCollisionsBuffer,
        m_numberOfCollisionsDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_numberOfCollisionsHostVisibleBuffer,
        m_numberOfCollisionsHostVisibleDeviceMemory);

    // Radix Time Map
    m_radixTimeMapDescriptorSetLayout = Compute::createDescriptorSetLayout(m_logicalDevice, CollisionsApplyerUtil::kRadixTimeMapNumberOfBindings);;
    m_radixTimeMapDescriptorPool = Compute::createDescriptorPool(m_logicalDevice, CollisionsApplyerUtil::kRadixTimeMapNumberOfBindings, 1);
    m_radixTimeMapPipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_radixTimeMapDescriptorSetLayout);
    m_radixTimeMapPipeline = Compute::createPipeline("src/GLSL/spv/CollisionsRadixSortTimeMap.spv", m_logicalDevice, m_radixTimeMapPipelineLayout);

    std::vector<Compute::BufferAndSize> radixTimeMapBufferAndSizes = {
        {m_computedCollisionsBuffer, computedCollisionsMemorySize},
        {m_radixSorter->m_dataBuffer, maxNumberOfAgents * CollisionsApplyerUtil::kMaxCollisionsPerAgent * sizeof(uint32_t)},
        {m_numberOfCollisionsBuffer, sizeof(uint32_t)}
    };

    m_radixTimeMapDescriptorSet = Compute::createDescriptorSet(
        m_logicalDevice,
        m_radixTimeMapDescriptorSetLayout,
        m_radixTimeMapDescriptorPool,
        radixTimeMapBufferAndSizes);

    // Radix Time Gather

    m_radixGatherDescriptorSetLayout = Compute::createDescriptorSetLayout(m_logicalDevice, CollisionsApplyerUtil::kRadixGatherNumberOfBindings);;
    m_radixGatherDescriptorPool = Compute::createDescriptorPool(m_logicalDevice, CollisionsApplyerUtil::kRadixGatherNumberOfBindings, 2);
    m_radixGatherPipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_radixGatherDescriptorSetLayout);
    m_radixGatherPipeline = Compute::createPipeline("src/GLSL/spv/CollisionsRadixSortGather.spv", m_logicalDevice, m_radixGatherPipelineLayout);

    std::vector<Compute::BufferAndSize> radixTimeGatherBufferAndSizes = {
        {m_computedCollisionsBuffer, computedCollisionsMemorySize},
        {m_radixSorter->m_dataBuffer, maxNumberOfAgents * CollisionsApplyerUtil::kMaxCollisionsPerAgent * sizeof(uint32_t)},
        {m_otherComputedCollisionsBuffer, computedCollisionsMemorySize},
        {m_numberOfCollisionsBuffer, sizeof(uint32_t)}
    };

    m_radixTimeGatherDescriptorSet = Compute::createDescriptorSet(
        m_logicalDevice,
        m_radixGatherDescriptorSetLayout,
        m_radixGatherDescriptorPool,
        radixTimeGatherBufferAndSizes);

    // Radix Agent Index Map

    // Radix Agent Index Gather

    std::vector<Compute::BufferAndSize> radixAgentIndexGatherBufferAndSizes = {
        {m_otherComputedCollisionsBuffer, computedCollisionsMemorySize},
        {m_radixSorter->m_dataBuffer, maxNumberOfAgents * CollisionsApplyerUtil::kMaxCollisionsPerAgent * sizeof(uint32_t)},
        {m_computedCollisionsBuffer, computedCollisionsMemorySize},
        {m_numberOfCollisionsBuffer, sizeof(uint32_t)}
    };

    m_radixAgentIndexGatherDescriptorSet = Compute::createDescriptorSet(
        m_logicalDevice,
        m_radixGatherDescriptorSetLayout,
        m_radixGatherDescriptorPool,
        radixAgentIndexGatherBufferAndSizes);

    // Commands

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute fence");
    }

    m_radixTimeMapCommandBuffer = VK_NULL_HANDLE;
    m_radixTimeGatherCommandBuffer = VK_NULL_HANDLE;
    createRadixSortCommandBuffers();
}

CollisionsApplyer::~CollisionsApplyer() {

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_radixTimeMapDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_logicalDevice, m_radixTimeMapDescriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_radixTimeMapPipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_radixTimeMapPipeline, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_radixGatherDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_logicalDevice, m_radixGatherDescriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_radixGatherPipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_radixGatherPipeline, nullptr);

    std::array<VkCommandBuffer, 2> commandBuffers = {
        m_radixTimeMapCommandBuffer,
        m_radixTimeGatherCommandBuffer
    };
    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, commandBuffers.size(), commandBuffers.data());

    vkFreeMemory(m_logicalDevice, m_computedCollisionsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_computedCollisionsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_otherComputedCollisionsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_otherComputedCollisionsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfCollisionsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfCollisionsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfCollisionsHostVisibleDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfCollisionsHostVisibleBuffer, nullptr);

    vkDestroyFence(m_logicalDevice, m_fence, nullptr);
}

void CollisionsApplyer::createRadixSortCommandBuffers() {
    std::array<VkCommandBuffer, 2> commandBuffers = {
        m_radixTimeMapCommandBuffer,
        m_radixTimeGatherCommandBuffer
    };
    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, commandBuffers.size(), commandBuffers.data());

    m_radixTimeMapCommandBuffer = CollisionsApplyerUtil::createRadixTimeMapCommandBuffer(
        m_logicalDevice,
        m_commandPool,
        m_radixTimeMapPipeline,
        m_radixTimeMapPipelineLayout,
        m_radixTimeMapDescriptorSet,
        m_numberOfCollisionsHostVisibleBuffer,
        m_numberOfCollisionsBuffer,
        m_currentNumberOfCollisions);

    m_radixTimeGatherCommandBuffer = CollisionsApplyerUtil::createRadixGatherCommandBuffer(
        m_logicalDevice,
        m_commandPool,
        m_radixGatherPipeline,
        m_radixGatherPipelineLayout,
        m_radixTimeGatherDescriptorSet,
        m_currentNumberOfCollisions);
}

void CollisionsApplyer::updateNumberOfElementsIfNecessary(uint32_t numberOfAgents, uint32_t numberOfCollisions) {
    if (m_currentNumberOfCollisions != numberOfCollisions) {
        m_currentNumberOfCollisions = numberOfCollisions;
        Buffer::writeHostVisible(&numberOfCollisions, m_numberOfCollisionsHostVisibleDeviceMemory, 0, sizeof(uint32_t), m_logicalDevice);
        createRadixSortCommandBuffers();
    }
}

void CollisionsApplyer::run(uint32_t numberOfAgents, uint32_t numberOfCollisions, float timeDelta) {
    updateNumberOfElementsIfNecessary(numberOfAgents, numberOfCollisions);
    Command::runAndWait(m_radixTimeMapCommandBuffer, m_fence, m_queue, m_logicalDevice);
    m_radixSorter->run(numberOfCollisions);
    Command::runAndWait(m_radixTimeGatherCommandBuffer, m_fence, m_queue, m_logicalDevice);
}
