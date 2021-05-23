#ifndef SCANNER_H
#define SCANNER_H

#include <Utils/MyGLM.h>

#include <vulkan/vulkan.h>

template<typename T>
class Scanner {

private:

    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    VkDeviceMemory m_dataDeviceMemory;

    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;
    VkPipelineLayout m_pipelineLayout;

    VkPipeline m_pipeline;
    VkPipeline m_addOffsetsPipeline;

    VkDescriptorSet m_descriptorSet;

    VkCommandBuffer m_commandBuffer;
    uint32_t m_currentNumberOfElements;

    VkFence m_fence;

    void createScanCommand(uint32_t numberOfElements);

    void createScanCommandIfNecessary(uint32_t numberOfElements);

public:

    VkBuffer m_dataBuffer;

    Scanner(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        uint32_t numberOfElements);

    virtual ~Scanner();

    void run(uint32_t numberOfElements);

    void recordCommand(VkCommandBuffer commandBuffer, uint32_t numberOfElements);
};

#endif
