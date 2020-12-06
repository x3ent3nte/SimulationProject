#ifndef REDUCE_H
#define REDUCE_H

#include <vulkan/vulkan.h>

class Reduce {
private:

    VkPhysicalDevice m_physicalDevice;
    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    VkBuffer m_bufferTwo;
    VkBuffer m_dataSizeBuffer;
    VkBuffer m_dataSizeBufferHostVisible;

    VkDeviceMemory m_bufferTwoMemory;
    VkDeviceMemory m_dataSizeBufferMemory;
    VkDeviceMemory m_dataSizeBufferMemoryHostVisible;

    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipline;

    VkDescriptorSet m_oneToTwo;
    VkDescriptorSet m_twoToOne;

    VkCommandBuffer m_setDataSizeCommandBuffer;

    uint32_t m_currentDataSize;

    void setDataSize(uint32_t);

    void runReduceCommand(uint32_t dataSize);

public:

    VkBuffer m_bufferOne;

    Reduce(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        uint32_t numberOfElements);

    virtual ~Reduce();

    VkBuffer run(uint32_t numberOfElements);
};

#endif
