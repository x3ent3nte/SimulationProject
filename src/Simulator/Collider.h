#ifndef COLLIDER_H
#define COLLIDER_H

#include <Simulator/Collision.h>
#include <Simulator/AgentSorter.h>
#include <Simulator/TimeAdvancer.h>
#include <Simulator/Impacter.h>
#include <Simulator/Scanner.h>
#include <Simulator/CollisionsApplyer.h>

#include <vulkan/vulkan.h>

#include <memory>

class Collider {

private:

    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;
    VkBuffer m_agentsBuffer;

    std::shared_ptr<AgentSorter> m_agentSorter;
    std::shared_ptr<Scanner<int32_t>> m_scanner;
    std::shared_ptr<TimeAdvancer> m_timeAdvancer;
    std::shared_ptr<Impacter> m_impacter;
    std::shared_ptr<CollisionsApplyer> m_applyer;

    VkBuffer m_collisionsBuffer;
    VkDeviceMemory m_collisionsDeviceMemory;

    VkBuffer m_timeDeltaBuffer;
    VkDeviceMemory m_timeDeltaDeviceMemory;

    VkBuffer m_timeDeltaBufferHostVisible;
    VkDeviceMemory m_timeDeltaDeviceMemoryHostVisible;

    VkBuffer m_numberOfElementsBuffer;
    VkDeviceMemory m_numberOfElementsDeviceMemory;

    VkBuffer m_numberOfElementsBufferHostVisible;
    VkDeviceMemory m_numberOfElementsDeviceMemoryHostVisible;

    VkBuffer m_numberOfCollisionsBufferHostVisible;
    VkDeviceMemory m_numberOfCollisionsDeviceMemoryHostVisible;

    VkDescriptorSetLayout m_detectionDescriptorSetLayout;
    VkDescriptorPool m_detectionDescriptorPool;
    VkPipelineLayout m_detectionPipelineLayout;
    VkPipeline m_detectionPipeline;
    VkDescriptorSet m_detectionDescriptorSet;

    VkDescriptorSetLayout m_scatterDescriptorSetLayout;
    VkDescriptorPool m_scatterDescriptorPool;
    VkPipelineLayout m_scatterPipelineLayout;
    VkPipeline m_scatterPipeline;
    VkDescriptorSet m_scatterDescriptorSet;

    VkCommandBuffer m_setNumberOfElementsCommandBuffer;
    VkCommandBuffer m_collisionDetectionCommandBuffer;
    VkCommandBuffer m_scatterCommandBuffer;

    VkFence m_fence;

    uint32_t m_currentNumberOfElements;

    void updateNumberOfElementsIfNecessary(uint32_t numberOfElements);
    void createDetectionCommandBuffer();
    void createScatterCommandBuffer();

public:

    Collider(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        VkBuffer agentsBuffer,
        uint32_t numberOfElements);

    virtual ~Collider();

    void run(float timeDelta, uint32_t numberOfElements);
};

#endif
