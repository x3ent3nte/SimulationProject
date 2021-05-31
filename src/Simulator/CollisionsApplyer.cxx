#include <Simulator/CollisionsApplyer.h>

#include <Simulator/ComputedCollision.h>
#include <Utils/Buffer.h>
#include <Utils/Command.h>
#include <Utils/Compute.h>

namespace CollisionsApplyerUtil {

    constexpr size_t xDim = 256;
    constexpr uint32_t kMaxCollisionsPerAgent = 10;
    constexpr size_t kNumberOfBindings = 5;

}

CollisionsApplyer::CollisionsApplyer(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    VkBuffer agentsBuffer,
    uint32_t maxNumberOfAgents) {

    m_currentNumberOfAgents = maxNumberOfAgents;

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
}

CollisionsApplyer::~CollisionsApplyer() {
    vkFreeMemory(m_logicalDevice, m_computedCollisionsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_computedCollisionsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_otherComputedCollisionsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_otherComputedCollisionsBuffer, nullptr);
}

void CollisionsApplyer::run(uint32_t numberOfAgents, uint32_t numberOfCollisions, float timeDelta) {

}
