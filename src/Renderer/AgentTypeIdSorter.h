#ifndef AGENT_TYPEID_SORTER_H
#define AGENT_TYPEID_SORTER_H

#include <Simulator/RadixSorter.h>

#include <vulkan/vulkan.h>

#include <memory>
#include <vector>

class AgentTypeIdSorter {

public:

    struct TypeIdIndex {
        uint32_t typeId;
        uint32_t index;
    };

    VkPhysicalDevice m_physicalDevice;
    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    VkDescriptorSetLayout m_mapDescriptorSetLayout;
    VkDescriptorPool m_mapDescriptorPool;
    VkPipelineLayout m_mapPipelineLayout;
    VkPipeline m_mapPipeline;

    VkDescriptorSetLayout m_gatherDescriptorSetLayout;
    VkDescriptorPool m_gatherDescriptorPool;
    VkPipelineLayout m_gatherPipelineLayout;
    VkPipeline m_gatherPipeline;

    VkDescriptorSetLayout m_updateDrawCommandsDescriptorSetLayout;
    VkDescriptorPool m_updateDrawCommandsDescriptorPool;
    VkPipelineLayout m_updateDrawCommandsPipelineLayout;
    VkPipeline m_updateDrawCommandsPipeline;

    std::shared_ptr<RadixSorter> m_radixSorter;

    AgentTypeIdSorter(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        uint32_t maxNumberOfElements,
        size_t descriptorPoolSize);

    virtual ~AgentTypeIdSorter();

private:

};

class AgentTypeIdSorterFunction {

public:

    AgentTypeIdSorterFunction(
        std::shared_ptr<AgentTypeIdSorter> agentTypeIdSorter,
        VkBuffer agentsIn,
        VkBuffer agentsOut,
        VkBuffer indirectDrawBuffer,
        uint32_t maxNumberOfAgents,
        uint32_t numberOfTypeIds);

    virtual ~AgentTypeIdSorterFunction();

    std::vector<AgentTypeIdSorter::TypeIdIndex> run(uint32_t numberOfElements);

private:

    std::shared_ptr<AgentTypeIdSorter> m_agentTypeIdSorter;

    uint32_t m_currentNumberOfElements;

    VkDescriptorSet m_mapDescriptorSet;
    VkDescriptorSet m_gatherDescriptorSet;
    VkDescriptorSet m_updateDrawCommandsDescriptorSet;

    VkBuffer m_numberOfElementsBuffer;
    VkDeviceMemory m_numberOfElementsDeviceMemory;

    VkBuffer m_numberOfElementsHostVisibleBuffer;
    VkDeviceMemory m_numberOfElementsHostVisibleDeviceMemory;

    VkCommandBuffer m_commandBuffer;

    VkFence m_fence;
};

#endif
