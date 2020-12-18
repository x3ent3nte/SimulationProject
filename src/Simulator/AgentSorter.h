#ifndef AGENT_SORTER_H
#define AGENT_SORTER_H

#include <Simulator/InsertionSorter.h>

#include <vulkan/vulkan.h>

#include <memory>

class AgentSorter {

private:

    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;
    VkBuffer m_agentsBuffer;

    std::shared_ptr<InsertionSorter> m_insertionSorter;

    VkBuffer m_otherAgentBuffer;
    VkDeviceMemory m_otherAgentDeviceMemory;

    VkBuffer m_timeDeltaBuffer;
    VkDeviceMemory m_timeDeltaDeviceMemory;

    VkBuffer m_timeDeltaBufferHostVisible;
    VkDeviceMemory m_timeDeltaDeviceMemoryHostVisible;

    VkBuffer m_numberOfElementsBuffer;
    VkDeviceMemory m_numberOfElementsDeviceMemory;

    VkBuffer m_numberOfElementsBufferHostVisible;
    VkDeviceMemory m_numberOfElementsDeviceMemoryHostVisible;

    VkDescriptorSetLayout m_mapAgentToXDescriptorSetLayout;
    VkDescriptorPool m_mapAgentToXDescriptorPool;
    VkPipelineLayout m_mapAgentToXPipelineLayout;
    VkPipeline m_mapAgentToXPipeline;
    VkDescriptorSet m_mapAgentToXDescriptorSet;

    VkDescriptorSetLayout m_mapXToAgentDescriptorSetLayout;
    VkDescriptorPool m_mapXToAgentDescriptorPool;
    VkPipelineLayout m_mapXToAgentPipelineLayout;
    VkPipeline m_mapXToAgentPipeline;
    VkDescriptorSet m_mapXToAgentDescriptorSet;

    VkCommandBuffer m_setNumberOfElementsCommandBuffer;
    VkCommandBuffer m_mapAgentToXCommandBuffer;
    VkCommandBuffer m_mapXToAgentCommandBuffer;

    VkFence m_fence;

    uint32_t m_currentNumberOfElements;

    void updateNumberOfElementsIfNecessary(uint32_t numberOfElements);
    void mapAgentToX(float timeDelta);
    void mapXToAgent();

    void createCommandBuffers(uint32_t numberOfElements);

public:

    AgentSorter(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        VkBuffer agentsBuffer,
        uint32_t numberOfElements);

    virtual ~AgentSorter();

    void run(float timeDelta, uint32_t numberOfElements);

};

#endif
