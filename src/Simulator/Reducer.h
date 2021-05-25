#ifndef REDUCER_H
#define REDUCER_H

#include <vulkan/vulkan.h>

class Reducer {

private:

    VkPhysicalDevice m_physicalDevice;
    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    VkDeviceMemory m_oneBufferMemory;

    VkBuffer m_twoBuffer;
    VkDeviceMemory m_twoBufferMemory;

    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipeline;

    VkDescriptorSet m_oneToTwo;
    VkDescriptorSet m_twoToOne;

    VkFence m_fence;

    VkCommandBuffer m_commandBuffer;

    uint32_t m_currentNumberOfElements;

    VkBuffer m_returnBuffer;

    void createCommandBuffer();
    void updateNumberOfElementsIfNecessary(uint32_t numberOfElements);

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
