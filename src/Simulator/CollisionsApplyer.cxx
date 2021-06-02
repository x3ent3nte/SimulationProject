#include <Simulator/CollisionsApplyer.h>

#include <Simulator/Agent.h>
#include <Simulator/ComputedCollision.h>
#include <Utils/Buffer.h>
#include <Utils/Command.h>
#include <Utils/Compute.h>

#include <array>
#include <vector>

namespace CollisionsApplyerUtil {

    constexpr size_t xDim = 256;
    constexpr uint32_t kMaxCollisionsPerAgent = 10;

    VkCommandBuffer createRadixTimeMapCommandBuffer(
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        std::shared_ptr<ShaderLambda> shaderLambda,
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

        uint32_t xGroups = ceil(((float) numberOfElements) / ((float) xDim));
        shaderLambda->record(commandBuffer, xGroups, 1, 1);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to end compute command buffer");
        }

        return commandBuffer;
    }

    VkCommandBuffer createRadixGatherCommandBuffer(
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        std::shared_ptr<ShaderLambda> shaderLambda,
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

        uint32_t xGroups = ceil(((float) numberOfElements) / ((float) xDim));
        shaderLambda->record(commandBuffer, xGroups, 1, 1);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to end compute command buffer");
        }

        return commandBuffer;
    }

    VkCommandBuffer createRadixAgentIndexMapCommandBuffer(
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        std::shared_ptr<ShaderLambda> shaderLambda,
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

        uint32_t xGroups = ceil(((float) numberOfElements) / ((float) xDim));
        shaderLambda->record(commandBuffer, xGroups, 1, 1);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to end compute command buffer");
        }

        return commandBuffer;
    }

    VkCommandBuffer createApplyCommandBuffer(
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        std::shared_ptr<ShaderLambda> shaderLambda,
        VkBuffer timeDeltaHostVisibleBuffer,
        VkBuffer timeDeltaBuffer,
        VkBuffer numberOfAgentsHostVisibleBuffer,
        VkBuffer numberOfAgentsBuffer,
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
        vkCmdCopyBuffer(commandBuffer, timeDeltaHostVisibleBuffer, timeDeltaBuffer, 1, &copyRegion);
        vkCmdCopyBuffer(commandBuffer, numberOfAgentsHostVisibleBuffer, numberOfAgentsBuffer, 1, &copyRegion);

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

        uint32_t xGroups = ceil(((float) numberOfElements) / ((float) xDim));
        shaderLambda->record(commandBuffer, xGroups, 1, 1);

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

    Buffer::createBufferWithData(
        &m_currentNumberOfCollisions,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        m_logicalDevice,
        m_commandPool,
        m_queue,
        m_numberOfAgentsBuffer,
        m_numberOfAgentsDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_numberOfAgentsHostVisibleBuffer,
        m_numberOfAgentsHostVisibleDeviceMemory);

    float zero = 0.0f;
    Buffer::createBufferWithData(
        &zero,
        sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        m_logicalDevice,
        m_commandPool,
        m_queue,
        m_timeDeltaBuffer,
        m_timeDeltaDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_timeDeltaHostVisibleBuffer,
        m_timeDeltaHostVisibleDeviceMemory);

    // Radix Time Map
    std::vector<Compute::BufferAndSize> radixTimeMapBufferAndSizes = {
        {m_computedCollisionsBuffer, computedCollisionsMemorySize},
        {m_radixSorter->m_dataBuffer, maxNumberOfAgents * CollisionsApplyerUtil::kMaxCollisionsPerAgent * sizeof(uint32_t)},
        {m_numberOfCollisionsBuffer, sizeof(uint32_t)}
    };

    auto radixTimeMapFn = std::make_shared<ShaderFunction>(m_logicalDevice, 3, "src/GLSL/spv/CollisionsRadixSortTimeMap.spv");
    auto radixTimeMapPool = std::make_shared<ShaderPool>(radixTimeMapFn, 1);
    m_radixTimeMapLambda = std::make_shared<ShaderLambda>(radixTimeMapPool, radixTimeMapBufferAndSizes);

    // Radix Time Gather
    std::vector<Compute::BufferAndSize> radixTimeGatherBufferAndSizes = {
        {m_computedCollisionsBuffer, computedCollisionsMemorySize},
        {m_radixSorter->m_dataBuffer, maxNumberOfAgents * CollisionsApplyerUtil::kMaxCollisionsPerAgent * sizeof(uint32_t)},
        {m_otherComputedCollisionsBuffer, computedCollisionsMemorySize},
        {m_numberOfCollisionsBuffer, sizeof(uint32_t)}
    };

    auto radixGatherFn = std::make_shared<ShaderFunction>(m_logicalDevice, 4, "src/GLSL/spv/CollisionsRadixSortGather.spv");
    auto radixGatherPool = std::make_shared<ShaderPool>(radixGatherFn, 2);
    m_radixTimeGatherLambda = std::make_shared<ShaderLambda>(radixGatherPool, radixTimeGatherBufferAndSizes);

    // Radix Agent Index Map
    std::vector<Compute::BufferAndSize> radixAgentIndexMapBufferAndSizes = {
        {m_otherComputedCollisionsBuffer, computedCollisionsMemorySize},
        {m_radixSorter->m_dataBuffer, maxNumberOfAgents * CollisionsApplyerUtil::kMaxCollisionsPerAgent * sizeof(uint32_t)},
        {m_numberOfCollisionsBuffer, sizeof(uint32_t)}
    };

    auto radixAgentIndexMapFn = std::make_shared<ShaderFunction>(m_logicalDevice, 3, "src/GLSL/spv/CollisionsRadixSortAgentIndexMap.spv");
    auto radixAgentIndexMapPool = std::make_shared<ShaderPool>(radixAgentIndexMapFn, 1);
    m_radixAgentIndexMapLambda = std::make_shared<ShaderLambda>(radixAgentIndexMapPool, radixAgentIndexMapBufferAndSizes);

    // Radix Agent Index Gather
    std::vector<Compute::BufferAndSize> radixAgentIndexGatherBufferAndSizes = {
        {m_otherComputedCollisionsBuffer, computedCollisionsMemorySize},
        {m_radixSorter->m_dataBuffer, maxNumberOfAgents * CollisionsApplyerUtil::kMaxCollisionsPerAgent * sizeof(uint32_t)},
        {m_computedCollisionsBuffer, computedCollisionsMemorySize},
        {m_numberOfCollisionsBuffer, sizeof(uint32_t)}
    };

    m_radixAgentIndexGatherLambda = std::make_shared<ShaderLambda>(radixGatherPool, radixAgentIndexGatherBufferAndSizes);

    // Apply
    std::vector<Compute::BufferAndSize> applyBufferAndSizes = {
        {agentsBuffer, maxNumberOfAgents * sizeof(Agent)},
        {m_computedCollisionsBuffer, computedCollisionsMemorySize},
        {m_timeDeltaBuffer, sizeof(float)},
        {m_numberOfAgentsBuffer, sizeof(uint32_t)},
        {m_numberOfCollisionsBuffer, sizeof(uint32_t)},
    };

    auto applyFn = std::make_shared<ShaderFunction>(m_logicalDevice, 5, "src/GLSL/spv/CollisionsApply.spv");
    auto applyPool = std::make_shared<ShaderPool>(applyFn, 1);
    m_applyLambda = std::make_shared<ShaderLambda>(applyPool, applyBufferAndSizes);

    // Commands

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute fence");
    }

    m_radixTimeMapCommandBuffer = VK_NULL_HANDLE;
    m_radixTimeGatherCommandBuffer = VK_NULL_HANDLE;
    m_radixAgentIndexMapCommandBuffer = VK_NULL_HANDLE;
    m_radixAgentIndexGatherCommandBuffer = VK_NULL_HANDLE;
    m_applyCommandBuffer = VK_NULL_HANDLE;
    createRadixSortCommandBuffers();
    createCollisionsApplyCommandBuffer();
}

CollisionsApplyer::~CollisionsApplyer() {

    std::array<VkCommandBuffer, 4> commandBuffers = {
        m_radixTimeMapCommandBuffer,
        m_radixTimeGatherCommandBuffer,
        m_radixAgentIndexMapCommandBuffer,
        m_radixAgentIndexGatherCommandBuffer
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

    vkFreeMemory(m_logicalDevice, m_numberOfAgentsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfAgentsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfAgentsHostVisibleDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfAgentsHostVisibleBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_timeDeltaDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_timeDeltaBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_timeDeltaHostVisibleDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_timeDeltaHostVisibleBuffer, nullptr);

    vkDestroyFence(m_logicalDevice, m_fence, nullptr);
}

void CollisionsApplyer::createRadixSortCommandBuffers() {
    std::array<VkCommandBuffer, 4> commandBuffers = {
        m_radixTimeMapCommandBuffer,
        m_radixTimeGatherCommandBuffer,
        m_radixAgentIndexMapCommandBuffer,
        m_radixAgentIndexGatherCommandBuffer
    };
    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, commandBuffers.size(), commandBuffers.data());

    m_radixTimeMapCommandBuffer = CollisionsApplyerUtil::createRadixTimeMapCommandBuffer(
        m_logicalDevice,
        m_commandPool,
        m_radixTimeMapLambda,
        m_numberOfCollisionsHostVisibleBuffer,
        m_numberOfCollisionsBuffer,
        m_currentNumberOfCollisions);

    m_radixTimeGatherCommandBuffer = CollisionsApplyerUtil::createRadixGatherCommandBuffer(
        m_logicalDevice,
        m_commandPool,
        m_radixTimeGatherLambda,
        m_currentNumberOfCollisions);

    m_radixAgentIndexMapCommandBuffer = CollisionsApplyerUtil::createRadixAgentIndexMapCommandBuffer(
        m_logicalDevice,
        m_commandPool,
        m_radixAgentIndexMapLambda,
        m_currentNumberOfCollisions);

    m_radixAgentIndexGatherCommandBuffer = CollisionsApplyerUtil::createRadixGatherCommandBuffer(
        m_logicalDevice,
        m_commandPool,
        m_radixAgentIndexGatherLambda,
        m_currentNumberOfCollisions);
}

void CollisionsApplyer::createCollisionsApplyCommandBuffer() {
    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &m_applyCommandBuffer);

    m_applyCommandBuffer = CollisionsApplyerUtil::createApplyCommandBuffer(
        m_logicalDevice,
        m_commandPool,
        m_applyLambda,
        m_timeDeltaHostVisibleBuffer,
        m_timeDeltaBuffer,
        m_numberOfAgentsHostVisibleBuffer,
        m_numberOfAgentsBuffer,
        m_currentNumberOfAgents);
}

void CollisionsApplyer::updateNumberOfElementsIfNecessary(uint32_t numberOfAgents, uint32_t numberOfCollisions) {
    if (m_currentNumberOfCollisions != numberOfCollisions) {
        m_currentNumberOfCollisions = numberOfCollisions;
        Buffer::writeHostVisible(&numberOfCollisions, m_numberOfCollisionsHostVisibleDeviceMemory, 0, sizeof(uint32_t), m_logicalDevice);
        createRadixSortCommandBuffers();
    }

    if (m_currentNumberOfAgents != numberOfAgents) {
        m_currentNumberOfAgents = numberOfAgents;
        Buffer::writeHostVisible(&numberOfAgents, m_numberOfAgentsHostVisibleDeviceMemory, 0, sizeof(uint32_t), m_logicalDevice);
        createCollisionsApplyCommandBuffer();
    }
}

void CollisionsApplyer::run(uint32_t numberOfAgents, uint32_t numberOfCollisions, float timeDelta) {
    updateNumberOfElementsIfNecessary(numberOfAgents, numberOfCollisions);

    Command::runAndWait(m_radixTimeMapCommandBuffer, m_fence, m_queue, m_logicalDevice);
    m_radixSorter->run(numberOfCollisions);
    Command::runAndWait(m_radixTimeGatherCommandBuffer, m_fence, m_queue, m_logicalDevice);

    Command::runAndWait(m_radixAgentIndexMapCommandBuffer, m_fence, m_queue, m_logicalDevice);
    m_radixSorter->run(numberOfCollisions);
    Command::runAndWait(m_radixAgentIndexGatherCommandBuffer, m_fence, m_queue, m_logicalDevice);

    Buffer::writeHostVisible(&timeDelta, m_timeDeltaHostVisibleDeviceMemory, 0, sizeof(float), m_logicalDevice);
    Command::runAndWait(m_applyCommandBuffer, m_fence, m_queue, m_logicalDevice);
}
