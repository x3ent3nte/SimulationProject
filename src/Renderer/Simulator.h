#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <vulkan/vulkan.h>

class Simulator {
private:

    VkQueue m_computeQueue;
    VkPipeline m_computePipeline;
    VkPipelineLayout m_computePipelineLayout;
    VkShaderModule m_computeShaderModule;

    VkCommandPool m_computeCommandPool;
    VkCommandBuffer m_computeCommandBuffer;
    VkFence m_computeFence;

    VkBuffer m_agentsBuffer;
    VkDeviceMemory m_agentsBufferMemory;

    VkBuffer m_positionsBuffer;
    VkDeviceMemory m_positionsBufferMemory;

    VkDescriptorPool m_computeDescriptorPool;
    VkDescriptorSet m_computeDescriptorSet;
    VkDescriptorSetLayout m_computeDescriptorSetLayout;

public:

    Simulator(VkPhysicalDevice physicalDevice, VkDevice logicalDevice);

    void compute(VkDevice logicalDevice);

    void cleanUp(VkDevice logicalDevice);
};

#endif
