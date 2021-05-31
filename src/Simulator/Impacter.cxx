#include <Simulator/Impacter.h>

#include <Simulator/Agent.h>
#include <Utils/Buffer.h>
#include <Utils/Compute.h>
#include <Utils/Command.h>
#include <Utils/MyGLM.h>

#include <vector>
#include <stdexcept>

namespace ImpacterUtil {

    constexpr size_t xDim = 256;
    constexpr uint32_t kMaxCollisionsPerAgent = 10;
    constexpr size_t kNumberOfBindings = 4;

    struct ComputedCollision {
        uint32_t agentIndex;
        float time;
        glm::vec3 velocityDelta;
    };

    VkCommandBuffer createImpactCommandBuffer(
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
} // end anonymous namespace

Impacter::Impacter(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    VkBuffer agentsBuffer,
    uint32_t numberOfElements)
    : m_logicalDevice(logicalDevice)
    , m_queue(queue)
    , m_commandPool(commandPool) {

    m_currentNumberOfElements = numberOfElements;
    const size_t collisionsMemorySize = numberOfElements * ImpacterUtil::kMaxCollisionsPerAgent * sizeof(Collision);
    const size_t computedCollisionsMemorySize = numberOfElements * 2 * ImpacterUtil::kMaxCollisionsPerAgent * sizeof(ImpacterUtil::ComputedCollision);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        collisionsMemorySize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_collisionBuffer,
        m_collisionDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        computedCollisionsMemorySize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_computedCollisionsBuffer,
        m_computedCollisionsDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        computedCollisionsMemorySize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_otherComputedCollisionsBuffer,
        m_otherComputedCollisionsDeviceMemory);

    Buffer::createBufferWithData(
        &numberOfElements,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        m_logicalDevice,
        m_commandPool,
        m_queue,
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

    m_descriptorSetLayout = Compute::createDescriptorSetLayout(logicalDevice, ImpacterUtil::kNumberOfBindings);
    m_descriptorPool = Compute::createDescriptorPool(logicalDevice, ImpacterUtil::kNumberOfBindings, 1);
    m_pipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_descriptorSetLayout);
    m_pipeline = Compute::createPipeline("src/GLSL/spv/CollisionsImpact.spv", m_logicalDevice, m_pipelineLayout);

    std::vector<Compute::BufferAndSize> bufferAndSizes = {
        {agentsBuffer, numberOfElements * sizeof(Agent)},
        {m_collisionBuffer, collisionsMemorySize},
        {m_computedCollisionsBuffer, computedCollisionsMemorySize},
        {m_numberOfElementsBuffer, sizeof(uint32_t)}
    };

    m_descriptorSet = Compute::createDescriptorSet(
        m_logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        bufferAndSizes);

    m_impactCommandBuffer = VK_NULL_HANDLE;
    createImpactCommandBuffer();

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute fence");
    }
}

Impacter::~Impacter() {
    vkFreeMemory(m_logicalDevice, m_collisionDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_collisionBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_computedCollisionsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_computedCollisionsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_otherComputedCollisionsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_otherComputedCollisionsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfElementsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfElementsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfElementsHostVisibleDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfElementsHostVisibleBuffer, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_descriptorSetLayout, nullptr);

    vkDestroyDescriptorPool(m_logicalDevice, m_descriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_pipeline, nullptr);

    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &m_impactCommandBuffer);

    vkDestroyFence(m_logicalDevice, m_fence, nullptr);
}

void Impacter::createImpactCommandBuffer() {
    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &m_impactCommandBuffer);
    m_impactCommandBuffer = ImpacterUtil::createImpactCommandBuffer(
        m_logicalDevice,
        m_commandPool,
        m_pipeline,
        m_pipelineLayout,
        m_descriptorSet,
        m_numberOfElementsHostVisibleBuffer,
        m_numberOfElementsBuffer,
        m_currentNumberOfElements);
}

void Impacter::updateNumberOfElementsIfNecessary(uint32_t numberOfElements) {
    if (m_currentNumberOfElements != numberOfElements) {
        m_currentNumberOfElements = numberOfElements;
        Buffer::writeHostVisible(&numberOfElements, m_numberOfElementsHostVisibleDeviceMemory, 0, sizeof(uint32_t), m_logicalDevice);
        createImpactCommandBuffer();
    }
}

void Impacter::run(uint32_t numberOfElements) {
    updateNumberOfElementsIfNecessary(numberOfElements);
    Command::runAndWait(m_impactCommandBuffer, m_fence, m_queue, m_logicalDevice);

    //Collision collisionCopy = collision;
    //Buffer::writeHostVisible(&collisionCopy, m_collisionHostVisibleDeviceMemory, 0, sizeof(Collision), m_logicalDevice);

    //Command::runAndWait(m_impactCommandBuffer, m_fence, m_queue, m_logicalDevice);
}
