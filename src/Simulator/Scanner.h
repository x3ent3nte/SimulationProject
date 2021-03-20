#ifndef SCANNER_H
#define SCANNER_H

#include <vulkan/vulkan.h>

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

    VkFence m_fence;

    void setInfo(uint32_t dataOffset, uint32_t offsetOffset, uint32_t numberOfElements);

    void addOffsets(uint32_t dataOffset, uint32_t offsetOffset, uint32_t numberOfElements);

    void runScanCommand(uint32_t dataOffset, uint32_t offsetOffset, uint32_t numberOfElements);

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
};

#endif
