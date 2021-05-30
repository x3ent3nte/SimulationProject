#include <Simulator/Collider.h>

#include <Simulator/Agent.h>
#include <Utils/Buffer.h>
#include <Utils/Compute.h>
#include <Utils/Timer.h>
#include <Utils/Command.h>
#include <Utils/MyGLM.h>

#include <array>
#include <stdexcept>
#include <iostream>

namespace ColliderUtil {
    constexpr size_t xDim = 256;
    constexpr size_t kNumberOfBindings = 5;
    constexpr uint32_t kMaxCollisionsPerAgent = 10;

    struct ComputedCollision {
        uint32_t agentIndex;
        float time;
        glm::vec3 velocityDelta;
    };

    VkDescriptorSet createDescriptorSet(
        VkDevice logicalDevice,
        VkDescriptorSetLayout descriptorSetLayout,
        VkDescriptorPool descriptorPool,
        VkBuffer agentsBuffer,
        VkBuffer collisionsBuffer,
        VkBuffer numberOfCollisionsBuffer,
        VkBuffer timeDeltaBuffer,
        VkBuffer numberOfElementsBuffer,
        uint32_t numberOfElements) {

        std::vector<Compute::BufferAndSize> bufferAndSizes = {
            {agentsBuffer, numberOfElements * sizeof(Agent)},
            {collisionsBuffer, numberOfElements * kMaxCollisionsPerAgent * sizeof(Collision)},
            {numberOfCollisionsBuffer, numberOfElements * sizeof(uint32_t)},
            {timeDeltaBuffer, sizeof(float)},
            {numberOfElementsBuffer, sizeof(uint32_t)}
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
        VkBuffer timeDeltaBuffer,
        VkBuffer timeDeltaHostVisibleBuffer,
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
        copyRegion.size = sizeof(float);
        vkCmdCopyBuffer(commandBuffer, timeDeltaHostVisibleBuffer, timeDeltaBuffer, 1, &copyRegion);

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
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
} // namespace anonymous

Collider::Collider(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    VkBuffer agentsBuffer,
    uint32_t numberOfElements)
    : m_logicalDevice(logicalDevice)
    , m_queue(queue)
    , m_commandPool(commandPool)
    , m_agentsBuffer(agentsBuffer) {

    m_currentNumberOfElements = numberOfElements;

    m_agentSorter = std::make_shared<AgentSorter>(
        physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        m_agentsBuffer,
        numberOfElements,
        true);

    m_reducer = std::make_shared<Reducer>(
        physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        numberOfElements);

    m_scanner = std::make_shared<Scanner<int32_t>>(
        physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        numberOfElements);

    m_timeAdvancer = std::make_shared<TimeAdvancer>(
        physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        m_agentsBuffer,
        numberOfElements);

    m_impacter = std::make_shared<Impacter>(
        physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        m_agentsBuffer,
        numberOfElements);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        numberOfElements * ColliderUtil::kMaxCollisionsPerAgent * sizeof(Collision),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_collisionsBuffer,
        m_collisionsDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        numberOfElements * ColliderUtil::kMaxCollisionsPerAgent * sizeof(Collision),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_compactedCollisionsBuffer,
        m_compactedCollisionsDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        numberOfElements * ColliderUtil::kMaxCollisionsPerAgent * sizeof(ColliderUtil::ComputedCollision),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_senderCollisionsBuffer,
        m_senderCollisionsDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        numberOfElements * ColliderUtil::kMaxCollisionsPerAgent * sizeof(ColliderUtil::ComputedCollision),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_receiverCollisionsBuffer,
        m_receiverCollisionsDeviceMemory);

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

    m_descriptorSetLayout = Compute::createDescriptorSetLayout(m_logicalDevice, ColliderUtil::kNumberOfBindings);
    m_descriptorPool = Compute::createDescriptorPool(m_logicalDevice, ColliderUtil::kNumberOfBindings, 1);
    m_pipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_descriptorSetLayout);
    m_pipeline = Compute::createPipeline("src/GLSL/spv/CollisionDetection.spv", m_logicalDevice, m_pipelineLayout);
    m_descriptorSet = ColliderUtil::createDescriptorSet(
        m_logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        m_agentsBuffer,
        m_collisionsBuffer,
        m_scanner->m_dataBuffer,
        m_timeDeltaBuffer,
        m_numberOfElementsBuffer,
        numberOfElements);

    m_collisionDetectionCommandBuffer = VK_NULL_HANDLE;
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

void Collider::createCommandBuffer(uint32_t numberOfElements) {
    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &m_collisionDetectionCommandBuffer);
    m_collisionDetectionCommandBuffer = ColliderUtil::createCommandBuffer(
        m_logicalDevice,
        m_commandPool,
        m_pipeline,
        m_pipelineLayout,
        m_descriptorSet,
        m_timeDeltaBuffer,
        m_timeDeltaBufferHostVisible,
        numberOfElements);
}

void Collider::updateNumberOfElementsIfNecessary(uint32_t numberOfElements) {
    if (m_currentNumberOfElements == numberOfElements) {
        return;
    }

    createCommandBuffer(numberOfElements);

    m_currentNumberOfElements = numberOfElements;

    Buffer::writeHostVisible(&numberOfElements, m_numberOfElementsDeviceMemoryHostVisible, 0, sizeof(uint32_t), m_logicalDevice);

    Command::runAndWait(m_setNumberOfElementsCommandBuffer, m_fence, m_queue, m_logicalDevice);
}

Collider::~Collider() {

    vkFreeMemory(m_logicalDevice, m_collisionsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_collisionsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_compactedCollisionsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_compactedCollisionsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_senderCollisionsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_senderCollisionsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_receiverCollisionsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_receiverCollisionsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_timeDeltaDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_timeDeltaBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_timeDeltaDeviceMemoryHostVisible, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_timeDeltaBufferHostVisible, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfElementsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfElementsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfElementsDeviceMemoryHostVisible, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfElementsBufferHostVisible, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_logicalDevice, m_descriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice,  m_pipeline, nullptr);

    std::array<VkCommandBuffer, 2> commandBuffers = {
        m_collisionDetectionCommandBuffer,
        m_setNumberOfElementsCommandBuffer};
    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, commandBuffers.size(), commandBuffers.data());

    vkDestroyFence(m_logicalDevice, m_fence, nullptr);
}

float Collider::computeNextStep(float timeDelta) {
    return timeDelta;
    //Collision earliestCollision = extractEarliestCollision(reduceResult);
    /*
    if (earliestCollision.time < timeDelta) {
        {
            //Timer timer("Advance Time");
            m_timeAdvancer->run(earliestCollision.time, m_currentNumberOfElements);
        }
        {
            //Timer timer("Impacter");
            m_impacter->run(earliestCollision);
        }
        return earliestCollision.time;
    } else {
        {
            //Timer timer("Advance Time Full");
            m_timeAdvancer->run(timeDelta, m_currentNumberOfElements);
        }
        return timeDelta;
    }
    */
}

void Collider::run(float timeDelta, uint32_t numberOfElements) {
    updateNumberOfElementsIfNecessary(numberOfElements);

    m_agentSorter->run(timeDelta, m_currentNumberOfElements);

    Buffer::writeHostVisible(&timeDelta, m_timeDeltaDeviceMemoryHostVisible, 0, sizeof(float), m_logicalDevice);
    Command::runAndWait(m_collisionDetectionCommandBuffer, m_fence, m_queue, m_logicalDevice);
    m_scanner->run(m_currentNumberOfElements);

    // scatter collisions
    // resolve collisions
    // sender evaluate
    // receiver evaluate

    m_timeAdvancer->run(timeDelta, m_currentNumberOfElements);

    /*

    int numberOfSteps = 0;
    while (timeDelta > 0.0f) {
        {
            //Timer timer("computeNextStep");
            float timeDepleted = computeNextStep(timeDelta);
            std::cout << "Time depleted= " << timeDepleted << "\n";
            timeDelta -= timeDepleted;
        }
        numberOfSteps += 1;
    }

    std::cout << "Number of Collider steps = " << numberOfSteps << "\n";
    */
}
