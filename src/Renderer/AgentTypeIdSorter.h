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

    VkDescriptorSetLayout m_mapDescriptorSet;
    VkDescriptorPool m_mapDescriptorPool;
    VkPipelineLayout m_mapPipelineLayout;
    VkPipeline m_mapPipeline;

    VkDescriptorSetLayout m_gatherDescriptorSet;
    VkDescriptorPool m_gatherDescriptorPool;
    VkPipelineLayout m_gatherPipelineLayout;
    VkPipeline m_gatherPipeline;

    VkDescriptorSetLayout m_offsetsDescriptorSet;
    VkDescriptorPool m_offsetsDescriptorPool;
    VkPipelineLayout m_offsetsPipelineLayout;
    VkPipeline m_offsetsPipeline;

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
        uint32_t maxNumberOfAgents);

    virtual ~AgentTypeIdSorterFunction();

    std::vector<AgentTypeIdSorter::TypeIdIndex> run(uint32_t numberOfElements);

private:

    std::shared_ptr<AgentTypeIdSorter> m_agentTypeIdSorter;
};

#endif
