#ifndef REPRODUCER_H
#define REPRODUCER_H

#include <vulkan/vulkan.h>

class Reproducer {

private:

    VkDevice m_logicalDevice;

    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;

    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipeline;

    VkDescriptorSet m_descriptorSet;

public:

    Reproducer(
        VkDevice logicalDevice,
        VkBuffer agentsInBuffer,
        VkBuffer addressesBuffer,
        VkBuffer agentsOutBuffer,
        uint32_t numberOfElements);

    virtual ~Reproducer();

    void recordCommand(VkCommandBuffer commandBuffer, uint32_t numberOfElements);
};

#endif
