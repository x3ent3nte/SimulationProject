#ifndef IMPACTER_H
#define IMPACTER_H

#include <Simulator/Collision.h>

#include <vulkan/vulkan.h>

class Impacter {

private:

    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    VkBuffer m_collisionBuffer;
    VkDeviceMemory m_collisionDeviceMemory;

    VkBuffer m_collisionHostVisibleBuffer;
    VkDeviceMemory m_collisionHostVisibleDeviceMemory;

    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipeline;
    VkDescriptorSet m_descriptorSet;

    VkCommandBuffer m_commandBuffer;

    VkFence m_fence;

public:

    Impacter(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        VkBuffer agentsBuffer,
        uint32_t numberOfElements);

    virtual ~Impacter();

    void run(const Collision& collision);

};

#endif
