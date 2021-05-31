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

    void createRadixSortCommandBuffers();
    void updateNumberOfElementsIfNecessary(uint32_t numberOfAgents, uint32_t numberOfCollisions);

    uint32_t m_currentNumberOfAgents;
    uint32_t m_currentNumberOfCollisions;

    std::shared_ptr<RadixSorter> m_radixSorter;

    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    VkDeviceMemory m_computedCollisionsDeviceMemory;

    VkBuffer m_otherComputedCollisionsBuffer;
    VkDeviceMemory m_otherComputedCollisionsDeviceMemory;

    VkBuffer m_numberOfCollisionsBuffer;
    VkDeviceMemory m_numberOfCollisionsDeviceMemory;

    VkBuffer m_numberOfCollisionsHostVisibleBuffer;
    VkDeviceMemory m_numberOfCollisionsHostVisibleDeviceMemory;

    VkDescriptorSetLayout m_radixTimeMapDescriptorSetLayout;
    VkDescriptorPool m_radixTimeMapDescriptorPool;
    VkPipelineLayout m_radixTimeMapPipelineLayout;
    VkPipeline m_radixTimeMapPipeline;
    VkDescriptorSet m_radixTimeMapDescriptorSet;

    VkDescriptorSetLayout m_radixGatherDescriptorSetLayout;
    VkDescriptorPool m_radixGatherDescriptorPool;
    VkPipelineLayout m_radixGatherPipelineLayout;
    VkPipeline m_radixGatherPipeline;
    VkDescriptorSet m_radixTimeGatherDescriptorSet;

    VkDescriptorSetLayout m_radixAgentIndexMapDescriptorSetLayout;
    VkDescriptorPool m_radixAgentIndexMapDescriptorPool;
    VkPipelineLayout m_radixAgentIndexMapPipelineLayout;
    VkPipeline m_radixAgentIndexMapPipeline;
    VkDescriptorSet m_radixAgentIndexMapDescriptorSet;

    VkDescriptorSet m_radixAgentIndexGatherDescriptorSet;

    VkCommandBuffer m_radixTimeMapCommandBuffer;
    VkCommandBuffer m_radixTimeGatherCommandBuffer;
    VkCommandBuffer m_radixAgentIndexMapCommandBuffer;
    VkCommandBuffer m_radixAgentIndexGatherCommandBuffer;

    VkFence m_fence;
};

#endif
