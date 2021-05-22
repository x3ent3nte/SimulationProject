#ifndef REDUCER_H
#define REDUCER_H

#include <vulkan/vulkan.h>

class Reducer {

private:

    VkPhysicalDevice m_physicalDevice;
    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    VkBuffer m_twoBuffer;
    VkBuffer m_numberOfElementsBuffer;
    VkBuffer m_numberOfElementsBufferHostVisible;

    VkDeviceMemory m_oneBufferMemory;
    VkDeviceMemory m_twoBufferMemory;
    VkDeviceMemory m_numberOfElementsBufferMemory;
    VkDeviceMemory m_numberOfElementsBufferMemoryHostVisible;

    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipeline;

    VkDescriptorSet m_oneToTwo;
    VkDescriptorSet m_twoToOne;

    VkFence m_fence;

    void runReduceCommand(uint32_t numberOfElements, VkDescriptorSet descriptorSet);

public:

    VkBuffer m_oneBuffer;

    Reducer(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        uint32_t numberOfElements);

    virtual ~Reducer();

    VkBuffer run(uint32_t numberOfElements);
};

#endif
