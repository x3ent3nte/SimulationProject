#ifndef COLLISIONS_APPLYER_H
#define COLLISIONS_APPLYER_H

#include <Simulator/RadixSorter.h>

#include <vulkan/vulkan.h>

#include <memory>
#include <cstdint>

class CollisionsApplyer {

public:

    CollisionsApplyer(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        VkBuffer agentsBuffer,
        uint32_t maxNumberOfAgents);

    virtual ~CollisionsApplyer();

    void run(uint32_t numberOfAgents, uint32_t numberOfCollisions, float timeDelta);

    VkBuffer m_computedCollisionsBuffer;

private:

    uint32_t m_currentNumberOfAgents;

    std::shared_ptr<RadixSorter> m_radixSorter;

    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    VkDeviceMemory m_computedCollisionsDeviceMemory;

    VkBuffer m_otherComputedCollisionsBuffer;
    VkDeviceMemory m_otherComputedCollisionsDeviceMemory;
};

#endif
