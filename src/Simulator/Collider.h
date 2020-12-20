#ifndef COLLIDER_H
#define COLLIDER_H

#include <Simulator/AgentSorter.h>
#include <Simulator/Reducer.h>

#include <vulkan/vulkan.h>

#include <memory>

class Collider {

private:

    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;
    VkBuffer m_agentsBuffer;

    std::shared_ptr<AgentSorter> m_agentSorter;
    std::shared_ptr<Reducer> m_reducer;

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

    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipeline;

    VkDescriptorSet m_descriptorSet;

    VkCommandBuffer m_commandBuffer;
    VkCommandBuffer m_setNumberOfElementsCommandBuffer;

    VkFence m_fence;

    uint32_t m_currentNumberOfElements;

    void updateNumberOfElementsIfNecessary(uint32_t numberOfElements);
    void createCommandBuffer(uint32_t numberOfElements);
    void runCollisionDetection(float timeDelta);

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
