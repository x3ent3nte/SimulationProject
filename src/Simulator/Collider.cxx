#include <Simulator/Collider.h>

#include <Simulator/ColliderUtil.h>
#include <Utils/Buffer.h>
#include <Utils/Compute.h>
#include <Utils/Timer.h>
#include <Utils/Command.h>

#include <array>
#include <stdexcept>
#include <iostream>

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
        numberOfElements * sizeof(Collision),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_collisionsHostVisibleBuffer,
        m_collisionsHostVisibleDeviceMemory);

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

    m_descriptorSetLayout = ColliderUtil::createDescriptorSetLayout(m_logicalDevice);
    m_descriptorPool = ColliderUtil::createDescriptorPool(m_logicalDevice, 1);
    m_pipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_descriptorSetLayout);
    m_pipeline = ColliderUtil::createPipeline(m_logicalDevice, m_pipelineLayout);
    m_descriptorSet = ColliderUtil::createDescriptorSet(
        m_logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        m_agentsBuffer,
        m_reducer->m_oneBuffer,
        m_timeDeltaBuffer,
        m_numberOfElementsBuffer,
        numberOfElements);

    m_commandBuffer = VK_NULL_HANDLE;
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
    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &m_commandBuffer);
    m_commandBuffer = ColliderUtil::createCommandBuffer(
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

void Collider::runCollisionDetection(float timeDelta) {

    Buffer::writeHostVisible(&timeDelta, m_timeDeltaDeviceMemoryHostVisible, 0, sizeof(float), m_logicalDevice);

    Command::runAndWait(m_commandBuffer, m_fence, m_queue, m_logicalDevice);
}

Collider::~Collider() {

    vkFreeMemory(m_logicalDevice, m_collisionsHostVisibleDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_collisionsHostVisibleBuffer, nullptr);

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
        m_commandBuffer,
        m_setNumberOfElementsCommandBuffer};
    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, commandBuffers.size(), commandBuffers.data());

    vkDestroyFence(m_logicalDevice, m_fence, nullptr);
}

Collision Collider::extractEarliestCollision(VkBuffer reduceResult) {
    size_t collisionsSize = m_currentNumberOfElements * sizeof(Collision);
    VkCommandBuffer copyCollisionsCommandBuffer = Buffer::recordCopyCommand(
        m_logicalDevice,
        m_commandPool,
        reduceResult,
        m_collisionsHostVisibleBuffer,
        collisionsSize);

    Command::runAndWait(copyCollisionsCommandBuffer, m_fence, m_queue, m_logicalDevice);

    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &copyCollisionsCommandBuffer);

    Collision earliestCollision;
    Buffer::readHostVisible(m_collisionsHostVisibleDeviceMemory, &earliestCollision, 0, sizeof(Collision), m_logicalDevice);

    return earliestCollision;
}

float Collider::computeNextStep(float timeDelta) {
    m_agentSorter->run(timeDelta, m_currentNumberOfElements);
    {
        //Timer timer("runCollisionDetection");
        runCollisionDetection(timeDelta);
    }
    Collision earliestCollision;
    {
        //Timer timer("Reduce Collisions");
        VkBuffer reduceResult = m_reducer->run(m_currentNumberOfElements);
        earliestCollision = extractEarliestCollision(reduceResult);
    }

    //std::cout << "Earliest collision one= " << earliestCollision.one << " two= " << earliestCollision.two
    //    << " time= " << earliestCollision.time << " timeDelta=" << timeDelta << " equal=" << (earliestCollision.time == timeDelta) << "\n";
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

}

void Collider::run(float timeDelta, uint32_t numberOfElements) {
    updateNumberOfElementsIfNecessary(numberOfElements);

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

    std::cout << "Number of Collider steps= " << numberOfSteps << "\n";
}
