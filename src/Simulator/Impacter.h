#ifndef IMPACTER_H
#define IMPACTER_H

#include <Simulator/Collision.h>

#include <vulkan/vulkan.h>

class Impacter {

private:

    uint32_t m_currentNumberOfElements;

    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    VkDeviceMemory m_collisionDeviceMemory;

    VkBuffer m_computedCollisionsBuffer;
    VkDeviceMemory m_computedCollisionsDeviceMemory;

    VkBuffer m_otherComputedCollisionsBuffer;
    VkDeviceMemory m_otherComputedCollisionsDeviceMemory;

    VkBuffer m_numberOfElementsBuffer;
    VkDeviceMemory m_numberOfElementsDeviceMemory;

    VkBuffer m_numberOfElementsHostVisibleBuffer;
    VkDeviceMemory m_numberOfElementsHostVisibleDeviceMemory;

    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipeline;
    VkDescriptorSet m_descriptorSet;

    VkCommandBuffer m_impactCommandBuffer;

    VkFence m_fence;

    void createImpactCommandBuffer();
    void updateNumberOfElementsIfNecessary(uint32_t numberOfElements);

public:

    VkBuffer m_collisionBuffer;

    Impacter(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        VkBuffer agentsBuffer,
        uint32_t numberOfElements);

    virtual ~Impacter();

    void run(uint32_t numberOfElements);
};

#endif
