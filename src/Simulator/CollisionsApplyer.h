#ifndef COLLISIONS_APPLYER_H
#define COLLISIONS_APPLYER_H

#include <vulkan/vulkan.h>

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

    void run(uint32_t numberOfAgents, uint32_t numberOfCollisions);

    VkBuffer m_computedCollisionsBuffer;

private:

    uint32_t m_currentNumberOfAgents;

    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    VkDeviceMemory m_computedCollisionsDeviceMemory;

    VkBuffer m_otherComputedCollisionsBuffer;
    VkDeviceMemory m_otherComputedCollisionsDeviceMemory;
};

#endif
