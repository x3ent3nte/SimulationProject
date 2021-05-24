#include <Simulator/Gravity.h>

#include <Simulator/Agent.h>
#include <Utils/Buffer.h>
#include <Utils/Command.h>
#include <Utils/Compute.h>
#include <Utils/MyGLM.h>

#include <stdexcept>
#include <array>
#include <vector>

namespace GravityUtil {

    constexpr size_t kXDim = 512;
    constexpr size_t kMapNumberOfBindings = 3;
    constexpr size_t kScatterNumberOfBindings = 4;
    constexpr size_t kGravityNumberOfBindings = 5;

    struct MassiveAgent {
        glm::vec3 position;
        float mass;
    };

} // namespace GravityUtil

Gravity::Gravity(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    VkBuffer agentsBuffer,
    uint32_t maxNumberOfElements) {

    m_logicalDevice = logicalDevice;
    m_queue = queue;
    m_commandPool = commandPool;
    m_currentNumberOfElements = maxNumberOfElements;

    m_scanner = std::make_shared<Scanner<int32_t>>(
        physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        maxNumberOfElements);

    // create buffers

    const size_t massiveAgentsMemorySize = maxNumberOfElements * sizeof(GravityUtil::MassiveAgent);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        massiveAgentsMemorySize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_massiveAgentsBuffer,
        m_massiveAgentsDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_numberOfElementsBuffer,
        m_numberOfElementsDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_numberOfElementsHostVisibleBuffer,
        m_numberOfElementsHostVisibleDeviceMemory);

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
        m_timeDeltaHostVisibleBuffer,
        m_timeDeltaHostVisibleDeviceMemory);

    // create pipeline
    m_mapDescriptorSetLayout = Compute::createDescriptorSetLayout(m_logicalDevice, GravityUtil::kMapNumberOfBindings);
    m_mapDescriptorPool = Compute::createDescriptorPool(m_logicalDevice, GravityUtil::kMapNumberOfBindings, 1);
    m_mapPipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_mapDescriptorSetLayout);
    m_mapPipeline = Compute::createPipeline("src/GLSL/spv/GravityMap.spv", m_logicalDevice, m_mapPipelineLayout);

    const size_t agentsMemorySize = maxNumberOfElements * sizeof(Agent);
    const size_t scanMemorySize = maxNumberOfElements * sizeof(uint32_t);

    const std::vector<Compute::BufferAndSize> mapBufferAndSizes = {
        {agentsBuffer, agentsMemorySize},
        {m_scanner->m_dataBuffer, scanMemorySize},
        {m_numberOfElementsBuffer, sizeof(uint32_t)}
    };

    m_mapDescriptorSet = Compute::createDescriptorSet(
        m_logicalDevice,
        m_mapDescriptorSetLayout,
        m_mapDescriptorPool,
        mapBufferAndSizes);

    m_scatterDescriptorSetLayout = Compute::createDescriptorSetLayout(m_logicalDevice, GravityUtil::kScatterNumberOfBindings);
    m_scatterDescriptorPool = Compute::createDescriptorPool(m_logicalDevice, GravityUtil::kScatterNumberOfBindings, 1);
    m_scatterPipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_scatterDescriptorSetLayout);
    m_scatterPipeline = Compute::createPipeline("src/GLSL/spv/GravityScatter.spv", m_logicalDevice, m_scatterPipelineLayout);

    const std::vector<Compute::BufferAndSize> scatterBufferAndSizes = {
        {agentsBuffer, agentsMemorySize},
        {m_scanner->m_dataBuffer, scanMemorySize},
        {m_massiveAgentsBuffer, massiveAgentsMemorySize},
        {m_numberOfElementsBuffer, sizeof(uint32_t)}
    };

    m_scatterDescriptorSet = Compute::createDescriptorSet(
        m_logicalDevice,
        m_scatterDescriptorSetLayout,
        m_scatterDescriptorPool,
        scatterBufferAndSizes);

    m_gravityDescriptorSetLayout = Compute::createDescriptorSetLayout(m_logicalDevice, GravityUtil::kGravityNumberOfBindings);
    m_gravityDescriptorPool = Compute::createDescriptorPool(m_logicalDevice, GravityUtil::kGravityNumberOfBindings, 1);
    m_gravityPipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_gravityDescriptorSetLayout);
    m_gravityPipeline = Compute::createPipeline("src/GLSL/spv/Gravity.spv", m_logicalDevice, m_gravityPipelineLayout);

    const std::vector<Compute::BufferAndSize> gravityBufferAndSizes = {
        {agentsBuffer, agentsMemorySize},
        {m_massiveAgentsBuffer, massiveAgentsMemorySize},
        {m_scanner->m_dataBuffer, scanMemorySize},
        {m_timeDeltaBuffer, sizeof(float)},
        {m_numberOfElementsBuffer, sizeof(uint32_t)}
    };

    m_gravityDescriptorSet = Compute::createDescriptorSet(
        m_logicalDevice,
        m_gravityDescriptorSetLayout,
        m_gravityDescriptorPool,
        gravityBufferAndSizes);

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute fence");
    }

    // create commands

    m_setNumberOfElementsCommandBuffer = Buffer::recordCopyCommand(
        m_logicalDevice,
        m_commandPool,
        m_numberOfElementsHostVisibleBuffer,
        m_numberOfElementsBuffer,
        sizeof(uint32_t));

    setNumberOfElements(maxNumberOfElements);
    createCommandBuffer();
}

Gravity::~Gravity() {
    std::array<VkCommandBuffer, 2> commandBuffers = {
        m_setNumberOfElementsCommandBuffer,
        m_commandBuffer};
    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, commandBuffers.size(), commandBuffers.data());

    // free buffers
    vkFreeMemory(m_logicalDevice, m_massiveAgentsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_massiveAgentsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfElementsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfElementsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfElementsHostVisibleDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfElementsHostVisibleBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_timeDeltaDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_timeDeltaBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_timeDeltaHostVisibleDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_timeDeltaHostVisibleBuffer, nullptr);

    // free pipeline
    vkDestroyDescriptorSetLayout(m_logicalDevice, m_mapDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_logicalDevice, m_mapDescriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_mapPipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_mapPipeline, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_scatterDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_logicalDevice, m_scatterDescriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_scatterPipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_scatterPipeline, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_gravityDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_logicalDevice, m_gravityDescriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_gravityPipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_gravityPipeline, nullptr);

    vkDestroyFence(m_logicalDevice, m_fence, nullptr);
}

void Gravity::setNumberOfElements(uint32_t numberOfElements) {
    m_currentNumberOfElements = numberOfElements;
    Buffer::writeHostVisible(&numberOfElements, m_numberOfElementsHostVisibleDeviceMemory, 0, sizeof(uint32_t), m_logicalDevice);
    Command::runAndWait(m_setNumberOfElementsCommandBuffer, m_fence, m_queue, m_logicalDevice);
}

void Gravity::createCommandBuffer() {
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

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = sizeof(float);
    vkCmdCopyBuffer(m_commandBuffer, m_timeDeltaHostVisibleBuffer, m_timeDeltaBuffer, 1, &copyRegion);

    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.pNext = nullptr;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        m_commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &memoryBarrier,
        0,
        nullptr,
        0,
        nullptr);

    const uint32_t xGroups = ceil(((float) m_currentNumberOfElements) / ((float) GravityUtil::kXDim));

    vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_mapPipeline);
    vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_mapPipelineLayout, 0, 1, &m_mapDescriptorSet, 0, nullptr);
    vkCmdDispatch(m_commandBuffer, xGroups, 1, 1);

    vkCmdPipelineBarrier(
        m_commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &memoryBarrier,
        0,
        nullptr,
        0,
        nullptr);

    m_scanner->recordCommand(m_commandBuffer, m_currentNumberOfElements);

    vkCmdPipelineBarrier(
        m_commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &memoryBarrier,
        0,
        nullptr,
        0,
        nullptr);

    vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_scatterPipeline);
    vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_scatterPipelineLayout, 0, 1, &m_scatterDescriptorSet, 0, nullptr);
    vkCmdDispatch(m_commandBuffer, xGroups, 1, 1);

    vkCmdPipelineBarrier(
        m_commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &memoryBarrier,
        0,
        nullptr,
        0,
        nullptr);

    vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_gravityPipeline);
    vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_gravityPipelineLayout, 0, 1, &m_gravityDescriptorSet, 0, nullptr);
    vkCmdDispatch(m_commandBuffer, xGroups, 1, 1);

    if (vkEndCommandBuffer(m_commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end compute command buffer");
    }
}

void Gravity::createCommandBufferIfNecessary(uint32_t numberOfElements) {
    if (m_currentNumberOfElements != numberOfElements) {
        setNumberOfElements(numberOfElements);
        vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &m_commandBuffer);
        createCommandBuffer();
    }
}

void Gravity::run(float timeDelta, uint32_t numberOfElements) {
    createCommandBufferIfNecessary(numberOfElements);
    Buffer::writeHostVisible(&timeDelta, m_timeDeltaHostVisibleDeviceMemory, 0, sizeof(float), m_logicalDevice);
    Command::runAndWait(m_commandBuffer, m_fence, m_queue, m_logicalDevice);
}
