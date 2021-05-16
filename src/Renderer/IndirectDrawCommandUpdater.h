#ifndef AGENT_TYPEID_SORTER_H
#define AGENT_TYPEID_SORTER_H

#include <Simulator/RadixSorter.h>

#include <vulkan/vulkan.h>

#include <memory>
#include <vector>

class IndirectDrawCommandUpdater {

public:

    struct TypeIdIndex {
        uint32_t typeId;
        uint32_t index;
    };

    const uint32_t m_numberOfDrawCommands;

    VkBuffer m_numberOfDrawCommandsBuffer;
    VkDeviceMemory m_numberOfDrawCommandsDeviceMemory;

    VkPhysicalDevice m_physicalDevice;
    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    VkDescriptorSetLayout m_resetDrawCommandsDescriptorSetLayout;
    VkDescriptorPool m_resetDrawCommandsDescriptorPool;
    VkPipelineLayout m_resetDrawCommandsPipelineLayout;
    VkPipeline m_resetDrawCommandsPipeline;

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

    VkDescriptorSetLayout m_updateInstanceCountForDrawCommandsDescriptorSetLayout;
    VkDescriptorPool m_updateInstanceCountForDrawCommandsDescriptorPool;
    VkPipelineLayout m_updateInstanceCountForDrawCommandsPipelineLayout;
    VkPipeline m_updateInstanceCountForDrawCommandsPipeline;

    std::shared_ptr<RadixSorter> m_radixSorter;

    IndirectDrawCommandUpdater(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        uint32_t maxNumberOfElements,
        uint32_t numberOfDrawCommands,
        size_t descriptorPoolSize);

    virtual ~IndirectDrawCommandUpdater();

private:

};

class IndirectDrawCommandUpdaterFunction {

public:

    IndirectDrawCommandUpdaterFunction(
        std::shared_ptr<IndirectDrawCommandUpdater> parent,
        VkBuffer agentsIn,
        VkBuffer agentsOut,
        VkBuffer indirectDrawBuffer,
        uint32_t maxNumberOfAgents);

    virtual ~IndirectDrawCommandUpdaterFunction();

    void run(uint32_t numberOfElements);

private:

    void runCommandAndWaitForFence(VkCommandBuffer commandBuffer);
    void destroyCommandBuffers();
    void setNumberOfElements(uint32_t numberOfElements);
    void createBeforeRadixSortCommand();
    void createAfterRadixSortCommand();
    void createCommandBuffers();
    void createCommandBuffersIfNecessary(uint32_t numberOfElements);

    std::shared_ptr<IndirectDrawCommandUpdater> m_parent;

    uint32_t m_currentNumberOfElements;

    VkDescriptorSet m_resetDrawCommandsDescriptorSet;
    VkDescriptorSet m_mapDescriptorSet;
    VkDescriptorSet m_gatherDescriptorSet;
    VkDescriptorSet m_updateDrawCommandsDescriptorSet;
    VkDescriptorSet m_updateInstanceCountForDrawCommandsDescriptorSet;

    VkBuffer m_numberOfElementsBuffer;
    VkDeviceMemory m_numberOfElementsDeviceMemory;

    VkBuffer m_numberOfElementsHostVisibleBuffer;
    VkDeviceMemory m_numberOfElementsHostVisibleDeviceMemory;

    VkCommandBuffer m_setNumberOfElementsCommandBuffer;
    VkCommandBuffer m_beforeRadixSortCommandBuffer;
    VkCommandBuffer m_afterRadixSortCommandBuffer;

    VkFence m_fence;
};

#endif
