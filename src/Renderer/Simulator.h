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

    VkBuffer m_computeBufferA;
    VkDeviceMemory m_computeBufferMemoryA;

    VkBuffer m_computeBufferB;
    VkDeviceMemory m_computeBufferMemoryB;

    VkBuffer m_computeBufferC;
    VkDeviceMemory m_computeBufferMemoryC;

    VkDescriptorPool m_computeDescriptorPool;
    VkDescriptorSet m_computeDescriptorSet;
    VkDescriptorSetLayout m_computeDescriptorSetLayout;

public:

    Simulator(VkPhysicalDevice physicalDevice, VkDevice logicalDevice);

    void compute(VkDevice logicalDevice);

    void cleanUp(VkDevice logicalDevice);
};

#endif
