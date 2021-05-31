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

    void createCollisionsApplyCommandBuffer();
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

    VkBuffer m_numberOfAgentsBuffer;
    VkDeviceMemory m_numberOfAgentsDeviceMemory;

    VkBuffer m_numberOfAgentsHostVisibleBuffer;
    VkDeviceMemory m_numberOfAgentsHostVisibleDeviceMemory;

    VkBuffer m_timeDeltaBuffer;
    VkDeviceMemory m_timeDeltaDeviceMemory;

    VkBuffer m_timeDeltaHostVisibleBuffer;
    VkDeviceMemory m_timeDeltaHostVisibleDeviceMemory;

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

    VkDescriptorSetLayout m_applyDescriptorSetLayout;
    VkDescriptorPool m_applyDescriptorPool;
    VkPipelineLayout m_applyPipelineLayout;
    VkPipeline m_applyPipeline;
    VkDescriptorSet m_applyDescriptorSet;

    VkCommandBuffer m_radixTimeMapCommandBuffer;
    VkCommandBuffer m_radixTimeGatherCommandBuffer;
    VkCommandBuffer m_radixAgentIndexMapCommandBuffer;
    VkCommandBuffer m_radixAgentIndexGatherCommandBuffer;
    VkCommandBuffer m_applyCommandBuffer;

    VkFence m_fence;
};

#endif
