#include <Renderer/IndirectDrawCommandUpdater.h>

#include <Simulator/Agent.h>
#include <Utils/Buffer.h>
#include <Utils/Compute.h>

#include <array>
#include <stdexcept>

namespace {
    constexpr size_t kXDim = 512;
    constexpr size_t kResetDrawCommandsNumberOfBindings = 2;
    constexpr size_t kMapNumberOfBindings = 3;
    constexpr size_t kGatherNumberOfBindings = 4;
    constexpr size_t kUpdateDrawNumberOfBindings = 3;
    constexpr size_t kUpdateInstanceCountForDrawNumberOfBindings = 2;
} // namespace anonymous

IndirectDrawCommandUpdater::IndirectDrawCommandUpdater(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t maxNumberOfElements,
    uint32_t numberOfDrawCommands,
    size_t descriptorPoolSize)
    : m_numberOfDrawCommands(numberOfDrawCommands) {

    m_physicalDevice = physicalDevice;
    m_logicalDevice = logicalDevice;
    m_queue = queue;
    m_commandPool = commandPool;

    m_resetDrawCommandsDescriptorSetLayout = Compute::createDescriptorSetLayout(m_logicalDevice, kResetDrawCommandsNumberOfBindings);
    m_resetDrawCommandsDescriptorPool = Compute::createDescriptorPool(m_logicalDevice, kResetDrawCommandsNumberOfBindings, descriptorPoolSize);
    m_resetDrawCommandsPipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_resetDrawCommandsDescriptorSetLayout);
    m_resetDrawCommandsPipeline = Compute::createPipeline("src/GLSL/spv/ResetIndirectDrawCommands.spv", m_logicalDevice, m_resetDrawCommandsPipelineLayout);

    m_mapDescriptorSetLayout = Compute::createDescriptorSetLayout(m_logicalDevice, kMapNumberOfBindings);
    m_mapDescriptorPool = Compute::createDescriptorPool(m_logicalDevice, kMapNumberOfBindings, descriptorPoolSize);
    m_mapPipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_mapDescriptorSetLayout);
    m_mapPipeline = Compute::createPipeline("src/GLSL/spv/AgentTypeIdSortMap.spv", m_logicalDevice, m_mapPipelineLayout);

    m_gatherDescriptorSetLayout = Compute::createDescriptorSetLayout(m_logicalDevice, kGatherNumberOfBindings);
    m_gatherDescriptorPool = Compute::createDescriptorPool(m_logicalDevice, kGatherNumberOfBindings, descriptorPoolSize);
    m_gatherPipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_gatherDescriptorSetLayout);
    m_gatherPipeline = Compute::createPipeline("src/GLSL/spv/AgentTypeIdSortGather.spv", m_logicalDevice, m_gatherPipelineLayout);

    m_updateDrawCommandsDescriptorSetLayout = Compute::createDescriptorSetLayout(m_logicalDevice, kUpdateDrawNumberOfBindings);
    m_updateDrawCommandsDescriptorPool = Compute::createDescriptorPool(m_logicalDevice, kUpdateDrawNumberOfBindings, descriptorPoolSize);
    m_updateDrawCommandsPipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_updateDrawCommandsDescriptorSetLayout);
    m_updateDrawCommandsPipeline = Compute::createPipeline("src/GLSL/spv/UpdateIndirectDrawCommands.spv", m_logicalDevice, m_updateDrawCommandsPipelineLayout);

    m_updateInstanceCountForDrawCommandsDescriptorSetLayout = Compute::createDescriptorSetLayout(m_logicalDevice, kUpdateInstanceCountForDrawNumberOfBindings);
    m_updateInstanceCountForDrawCommandsDescriptorPool = Compute::createDescriptorPool(m_logicalDevice, kUpdateInstanceCountForDrawNumberOfBindings, descriptorPoolSize);
    m_updateInstanceCountForDrawCommandsPipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_updateInstanceCountForDrawCommandsDescriptorSetLayout);
    m_updateInstanceCountForDrawCommandsPipeline = Compute::createPipeline("src/GLSL/spv/UpdateInstanceCountForIndirectDrawCommands.spv", m_logicalDevice, m_updateInstanceCountForDrawCommandsPipelineLayout);

    uint32_t numberOfDrawCommandsCopy = m_numberOfDrawCommands;
    Buffer::createBufferWithData(
        &numberOfDrawCommandsCopy,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        m_physicalDevice,
        m_logicalDevice,
        m_commandPool,
        m_queue,
        m_numberOfDrawCommandsBuffer,
        m_numberOfDrawCommandsDeviceMemory);

    m_radixSorter = std::make_shared<RadixSorter>(
        physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        maxNumberOfElements);
}

IndirectDrawCommandUpdater::~IndirectDrawCommandUpdater() {

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_resetDrawCommandsDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_logicalDevice, m_resetDrawCommandsDescriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_resetDrawCommandsPipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_resetDrawCommandsPipeline, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_mapDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_logicalDevice, m_mapDescriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_mapPipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_mapPipeline, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_gatherDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_logicalDevice, m_gatherDescriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_gatherPipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_gatherPipeline, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_updateDrawCommandsDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_logicalDevice, m_updateDrawCommandsDescriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_updateDrawCommandsPipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_updateDrawCommandsPipeline, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_updateInstanceCountForDrawCommandsDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_logicalDevice, m_updateInstanceCountForDrawCommandsDescriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_updateInstanceCountForDrawCommandsPipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_updateInstanceCountForDrawCommandsPipeline, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfDrawCommandsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfDrawCommandsBuffer, nullptr);
}

IndirectDrawCommandUpdaterFunction::IndirectDrawCommandUpdaterFunction(
    std::shared_ptr<IndirectDrawCommandUpdater> parent,
    VkBuffer agentsIn,
    VkBuffer agentsOut,
    VkBuffer indirectDrawBuffer,
    uint32_t maxNumberOfAgents) {

    m_parent = parent;

    Buffer::createBuffer(
        m_parent->m_physicalDevice,
        m_parent->m_logicalDevice,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_numberOfElementsBuffer,
        m_numberOfElementsDeviceMemory);

    Buffer::createBuffer(
        m_parent->m_physicalDevice,
        m_parent->m_logicalDevice,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_numberOfElementsHostVisibleBuffer,
        m_numberOfElementsHostVisibleDeviceMemory);

    const size_t agentsMemorySize = maxNumberOfAgents * sizeof(AgentRenderInfo);
    const size_t valueAndIndexesMemorySize = maxNumberOfAgents * sizeof(RadixSorter::ValueAndIndex);
    const size_t drawCommandsMemorySize = m_parent->m_numberOfDrawCommands * sizeof(VkDrawIndexedIndirectCommand);

    const std::vector<Compute::BufferAndSize> resetDrawCommandsBufferAndSizes = {
        {indirectDrawBuffer, drawCommandsMemorySize},
        {m_parent->m_numberOfDrawCommandsBuffer, sizeof(uint32_t)}
    };

    m_resetDrawCommandsDescriptorSet = Compute::createDescriptorSet(
        m_parent->m_logicalDevice,
        m_parent->m_resetDrawCommandsDescriptorSetLayout,
        m_parent->m_resetDrawCommandsDescriptorPool,
        resetDrawCommandsBufferAndSizes);

    const std::vector<Compute::BufferAndSize> mapBufferAndSizes = {
        {agentsIn, agentsMemorySize},
        {m_parent->m_radixSorter->m_dataBuffer, valueAndIndexesMemorySize},
        {m_numberOfElementsBuffer, sizeof(uint32_t)}
    };

    m_mapDescriptorSet = Compute::createDescriptorSet(
        m_parent->m_logicalDevice,
        m_parent->m_mapDescriptorSetLayout,
        m_parent->m_mapDescriptorPool,
        mapBufferAndSizes);

    const std::vector<Compute::BufferAndSize> gatherBufferAndSizes = {
        {agentsIn, agentsMemorySize},
        {m_parent->m_radixSorter->m_dataBuffer, valueAndIndexesMemorySize},
        {agentsOut, agentsMemorySize},
        {m_numberOfElementsBuffer, sizeof(uint32_t)}
    };

    m_gatherDescriptorSet = Compute::createDescriptorSet(
        m_parent->m_logicalDevice,
        m_parent->m_gatherDescriptorSetLayout,
        m_parent->m_gatherDescriptorPool,
        gatherBufferAndSizes);

    const std::vector<Compute::BufferAndSize> updateDrawCommandsBufferAndSizes = {
        {agentsOut, agentsMemorySize},
        {indirectDrawBuffer, drawCommandsMemorySize},
        {m_numberOfElementsBuffer, sizeof(uint32_t)}
    };

    m_updateDrawCommandsDescriptorSet = Compute::createDescriptorSet(
        m_parent->m_logicalDevice,
        m_parent->m_updateDrawCommandsDescriptorSetLayout,
        m_parent->m_updateDrawCommandsDescriptorPool,
        updateDrawCommandsBufferAndSizes);

    const std::vector<Compute::BufferAndSize> updateInstanceCountForDrawCommandsBufferAndSizes = {
        {indirectDrawBuffer, drawCommandsMemorySize},
        {m_parent->m_numberOfDrawCommandsBuffer, sizeof(uint32_t)}
    };

    m_updateInstanceCountForDrawCommandsDescriptorSet = Compute::createDescriptorSet(
        m_parent->m_logicalDevice,
        m_parent->m_updateInstanceCountForDrawCommandsDescriptorSetLayout,
        m_parent->m_updateInstanceCountForDrawCommandsDescriptorPool,
        updateInstanceCountForDrawCommandsBufferAndSizes);

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    m_setNumberOfElementsCommandBuffer = Buffer::recordCopyCommand(
        m_parent->m_logicalDevice,
        m_parent->m_commandPool,
        m_numberOfElementsHostVisibleBuffer,
        m_numberOfElementsBuffer,
        sizeof(uint32_t));

    if (vkCreateFence(m_parent->m_logicalDevice, &fenceCreateInfo, nullptr, &m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute fence");
    }

    setNumberOfElements(maxNumberOfAgents);
    createCommandBuffers();
}

IndirectDrawCommandUpdaterFunction::~IndirectDrawCommandUpdaterFunction() {
    std::array<VkCommandBuffer, 1> commandBuffers = {
        m_setNumberOfElementsCommandBuffer};
    vkFreeCommandBuffers(m_parent->m_logicalDevice, m_parent->m_commandPool, commandBuffers.size(), commandBuffers.data());
    destroyCommandBuffers();

    vkFreeMemory(m_parent->m_logicalDevice, m_numberOfElementsDeviceMemory, nullptr);
    vkDestroyBuffer(m_parent->m_logicalDevice, m_numberOfElementsBuffer, nullptr);

    vkFreeMemory(m_parent->m_logicalDevice, m_numberOfElementsHostVisibleDeviceMemory, nullptr);
    vkDestroyBuffer(m_parent->m_logicalDevice, m_numberOfElementsHostVisibleBuffer, nullptr);

    vkDestroyFence(m_parent->m_logicalDevice, m_fence, nullptr);
}

void IndirectDrawCommandUpdaterFunction::runCommandAndWaitForFence(VkCommandBuffer commandBuffer) {
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkResetFences(m_parent->m_logicalDevice, 1, &m_fence);

    if (vkQueueSubmit(m_parent->m_queue, 1, &submitInfo, m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit RadixSorter command buffer");
    }
    vkWaitForFences(m_parent->m_logicalDevice, 1, &m_fence, VK_TRUE, UINT64_MAX);
}

void IndirectDrawCommandUpdaterFunction::destroyCommandBuffers() {
    std::array<VkCommandBuffer, 2> commandBuffers = {
        m_beforeRadixSortCommandBuffer,
        m_afterRadixSortCommandBuffer};
    vkFreeCommandBuffers(m_parent->m_logicalDevice, m_parent->m_commandPool, commandBuffers.size(), commandBuffers.data());
}

void IndirectDrawCommandUpdaterFunction::setNumberOfElements(uint32_t numberOfElements) {
    m_currentNumberOfElements = numberOfElements;

    void* dataMap;
    vkMapMemory(m_parent->m_logicalDevice, m_numberOfElementsHostVisibleDeviceMemory, 0, sizeof(uint32_t), 0, &dataMap);
    uint32_t numberOfElementsCopy = numberOfElements;
    memcpy(dataMap, &numberOfElementsCopy, sizeof(uint32_t));
    vkUnmapMemory(m_parent->m_logicalDevice, m_numberOfElementsHostVisibleDeviceMemory);

    runCommandAndWaitForFence(m_setNumberOfElementsCommandBuffer);
}

void IndirectDrawCommandUpdaterFunction::createBeforeRadixSortCommand() {
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = m_parent->m_commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(m_parent->m_logicalDevice, &commandBufferAllocateInfo, &m_beforeRadixSortCommandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute command buffer");
    }

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    if (vkBeginCommandBuffer(m_beforeRadixSortCommandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin compute command buffer");
    }

    const uint32_t mapXGroups = ceil(((float) m_currentNumberOfElements) / ((float) kXDim));

    vkCmdBindPipeline(m_beforeRadixSortCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_parent->m_mapPipeline);
    vkCmdBindDescriptorSets(m_beforeRadixSortCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_parent->m_mapPipelineLayout, 0, 1, &m_mapDescriptorSet, 0, nullptr);
    vkCmdDispatch(m_beforeRadixSortCommandBuffer, mapXGroups, 1, 1);

    const uint32_t resetDrawCommandsXGroups = ceil(((float) m_parent->m_numberOfDrawCommands) / ((float) kXDim));

    vkCmdBindPipeline(m_beforeRadixSortCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_parent->m_resetDrawCommandsPipeline);
    vkCmdBindDescriptorSets(m_beforeRadixSortCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_parent->m_resetDrawCommandsPipelineLayout, 0, 1, &m_resetDrawCommandsDescriptorSet, 0, nullptr);
    vkCmdDispatch(m_beforeRadixSortCommandBuffer, resetDrawCommandsXGroups, 1, 1);

    if (vkEndCommandBuffer(m_beforeRadixSortCommandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end compute command buffer");
    }
}

void IndirectDrawCommandUpdaterFunction::createAfterRadixSortCommand() {
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = m_parent->m_commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(m_parent->m_logicalDevice, &commandBufferAllocateInfo, &m_afterRadixSortCommandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute command buffer");
    }

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    if (vkBeginCommandBuffer(m_afterRadixSortCommandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin compute command buffer");
    }

    const uint32_t gatherXGroups = ceil(((float) m_currentNumberOfElements) / ((float) kXDim));

    vkCmdBindPipeline(m_afterRadixSortCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_parent->m_gatherPipeline);
    vkCmdBindDescriptorSets(m_afterRadixSortCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_parent->m_gatherPipelineLayout, 0, 1, &m_gatherDescriptorSet, 0, nullptr);
    vkCmdDispatch(m_afterRadixSortCommandBuffer, gatherXGroups, 1, 1);

    VkMemoryBarrier memoryBarrier = {};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.pNext = nullptr;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        m_afterRadixSortCommandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &memoryBarrier,
        0,
        nullptr,
        0,
        nullptr);

    const uint32_t updateDrawCommandsXGroups = ceil(((float) m_parent->m_numberOfDrawCommands) / ((float) kXDim));

    vkCmdBindPipeline(m_afterRadixSortCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_parent->m_updateDrawCommandsPipeline);
    vkCmdBindDescriptorSets(m_afterRadixSortCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_parent->m_updateDrawCommandsPipelineLayout, 0, 1, &m_updateDrawCommandsDescriptorSet, 0, nullptr);
    vkCmdDispatch(m_afterRadixSortCommandBuffer, gatherXGroups, 1, 1);

    vkCmdPipelineBarrier(
        m_afterRadixSortCommandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &memoryBarrier,
        0,
        nullptr,
        0,
        nullptr);

    vkCmdBindPipeline(m_afterRadixSortCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_parent->m_updateInstanceCountForDrawCommandsPipeline);
    vkCmdBindDescriptorSets(m_afterRadixSortCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_parent->m_updateInstanceCountForDrawCommandsPipelineLayout, 0, 1, &m_updateInstanceCountForDrawCommandsDescriptorSet, 0, nullptr);
    vkCmdDispatch(m_afterRadixSortCommandBuffer, updateDrawCommandsXGroups, 1, 1);

    if (vkEndCommandBuffer(m_afterRadixSortCommandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end compute command buffer");
    }
}

void IndirectDrawCommandUpdaterFunction::createCommandBuffers() {
    createBeforeRadixSortCommand();
    createAfterRadixSortCommand();
}

void IndirectDrawCommandUpdaterFunction::createCommandBuffersIfNecessary(uint32_t numberOfElements) {
    if (numberOfElements != m_currentNumberOfElements) {
        destroyCommandBuffers();
        setNumberOfElements(numberOfElements);
        createCommandBuffers();
    }
}

std::vector<IndirectDrawCommandUpdater::TypeIdIndex> IndirectDrawCommandUpdaterFunction::run(uint32_t numberOfElements) {
    createCommandBuffersIfNecessary(numberOfElements);
    runCommandAndWaitForFence(m_beforeRadixSortCommandBuffer);
    m_parent->m_radixSorter->run(numberOfElements);
    runCommandAndWaitForFence(m_afterRadixSortCommandBuffer);

    return {
        {0, 0},
        {1, numberOfElements / 2}
    };
}
