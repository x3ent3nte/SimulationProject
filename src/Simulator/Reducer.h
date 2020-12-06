#ifndef REDUCE_H
#define REDUCE_H

#include <vulkan/vulkan.h>

class Reducer {
private:

    VkPhysicalDevice m_physicalDevice;
    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    VkBuffer m_twoBuffer;
    VkBuffer m_dataSizeBuffer;
    VkBuffer m_dataSizeBufferHostVisible;

    VkDeviceMemory m_oneBufferMemory;
    VkDeviceMemory m_twoBufferMemory;
    VkDeviceMemory m_dataSizeBufferMemory;
    VkDeviceMemory m_dataSizeBufferMemoryHostVisible;

    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipeline;

    VkDescriptorSet m_oneToTwo;
    VkDescriptorSet m_twoToOne;

    VkFence m_fence;

    void setDataSize(uint32_t);

    void runReduceCommand(uint32_t dataSize, VkDescriptorSet descriptorSet);

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
