#include <Renderer/AgentTypeIdSorter.h>

#include <Simulator/Agent.h>
#include <Utils/Buffer.h>
#include <Utils/Compute.h>

namespace {
    constexpr size_t kXDim = 512;
    constexpr size_t kMapNumberOfBindings = 3;
    constexpr size_t kGatherNumberOfBindings = 4;
    constexpr size_t kUpdateDrawNumberOfBindings = 3;
} // namespace anonymous

AgentTypeIdSorter::AgentTypeIdSorter(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t maxNumberOfElements,
    size_t descriptorPoolSize) {

    m_physicalDevice = physicalDevice;
    m_logicalDevice = logicalDevice;
    m_queue = queue;
    m_commandPool = commandPool;

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

    m_radixSorter = std::make_shared<RadixSorter>(
        physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        maxNumberOfElements);
}

AgentTypeIdSorter::~AgentTypeIdSorter() {
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
}

AgentTypeIdSorterFunction::AgentTypeIdSorterFunction(
    std::shared_ptr<AgentTypeIdSorter> agentTypeIdSorter,
    VkBuffer agentsIn,
    VkBuffer agentsOut,
    VkBuffer indirectDrawBuffer,
    uint32_t maxNumberOfAgents,
    uint32_t numberOfTypeIds) {

    m_agentTypeIdSorter = agentTypeIdSorter;

    m_currentNumberOfElements = maxNumberOfAgents;

    Buffer::createBuffer(
        m_agentTypeIdSorter->m_physicalDevice,
        m_agentTypeIdSorter->m_logicalDevice,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_numberOfElementsBuffer,
        m_numberOfElementsDeviceMemory);

    Buffer::createBuffer(
        m_agentTypeIdSorter->m_physicalDevice,
        m_agentTypeIdSorter->m_logicalDevice,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_numberOfElementsHostVisibleBuffer,
        m_numberOfElementsHostVisibleDeviceMemory);

    const size_t agentsMemorySize = maxNumberOfAgents * sizeof(AgentRenderInfo);
    const size_t valueAndIndexesSize = maxNumberOfAgents * sizeof(RadixSorter::ValueAndIndex);

    const std::vector<Compute::BufferAndSize> mapBufferAndSizes = {
        {agentsIn, agentsMemorySize},
        {m_agentTypeIdSorter->m_radixSorter->m_dataBuffer, valueAndIndexesSize},
        {m_numberOfElementsBuffer, sizeof(uint32_t)}
    };

    m_mapDescriptorSet = Compute::createDescriptorSet(
        m_agentTypeIdSorter->m_logicalDevice,
        m_agentTypeIdSorter->m_mapDescriptorSetLayout,
        m_agentTypeIdSorter->m_mapDescriptorPool,
        mapBufferAndSizes);

    const std::vector<Compute::BufferAndSize> gatherBufferAndSizes = {
        {agentsIn, agentsMemorySize},
        {m_agentTypeIdSorter->m_radixSorter->m_dataBuffer, valueAndIndexesSize},
        {agentsOut, agentsMemorySize},
        {m_numberOfElementsBuffer, sizeof(uint32_t)}
    };

    m_gatherDescriptorSet = Compute::createDescriptorSet(
        m_agentTypeIdSorter->m_logicalDevice,
        m_agentTypeIdSorter->m_gatherDescriptorSetLayout,
        m_agentTypeIdSorter->m_gatherDescriptorPool,
        gatherBufferAndSizes);

    const std::vector<Compute::BufferAndSize> updateDrawCommandsBufferAndSizes = {
        {agentsOut, agentsMemorySize},
        {indirectDrawBuffer, numberOfTypeIds * sizeof(VkDrawIndexedIndirectCommand)},
        {m_numberOfElementsBuffer, sizeof(uint32_t)}
    };

    m_updateDrawCommandsDescriptorSet = Compute::createDescriptorSet(
        m_agentTypeIdSorter->m_logicalDevice,
        m_agentTypeIdSorter->m_updateDrawCommandsDescriptorSetLayout,
        m_agentTypeIdSorter->m_updateDrawCommandsDescriptorPool,
        updateDrawCommandsBufferAndSizes);

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateFence(m_agentTypeIdSorter->m_logicalDevice, &fenceCreateInfo, nullptr, &m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute fence");
    }
}

AgentTypeIdSorterFunction::~AgentTypeIdSorterFunction() {
    vkFreeMemory(m_agentTypeIdSorter->m_logicalDevice, m_numberOfElementsDeviceMemory, nullptr);
    vkDestroyBuffer(m_agentTypeIdSorter->m_logicalDevice, m_numberOfElementsBuffer, nullptr);

    vkFreeMemory(m_agentTypeIdSorter->m_logicalDevice, m_numberOfElementsHostVisibleDeviceMemory, nullptr);
    vkDestroyBuffer(m_agentTypeIdSorter->m_logicalDevice, m_numberOfElementsHostVisibleBuffer, nullptr);

    vkDestroyFence(m_agentTypeIdSorter->m_logicalDevice, m_fence, nullptr);
}

std::vector<AgentTypeIdSorter::TypeIdIndex> AgentTypeIdSorterFunction::run(uint32_t numberOfElements) {
    m_agentTypeIdSorter->m_radixSorter->run(numberOfElements);
    return {
        {0, 0},
        {1, numberOfElements / 2}
    };
}
