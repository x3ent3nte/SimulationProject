#ifndef INSERTION_SORT_H
#define INSERTION_SORT_H

#include <Renderer/InsertionSortUtil.h>

#include <vulkan/vulkan.h>

class InsertionSort {

private:

    VkBuffer m_valueAndIndexBuffer;
    VkBuffer m_infoBuffer;

    VkDeviceMemory m_valueAndIndexBufferMemory;
    VkDeviceMemory m_infoBufferMemory;

    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;
    VkDescriptorSet m_descriptorSet;

    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipeline;

    VkCommandBuffer m_commandBuffer;
    VkFence m_fence;

public:

    InsertionSort(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool);

    virtual ~InsertionSort() = default;

    void run();

    void cleanUp(VkDevice logicalDevice, VkCommandPool commandPool);
};

#endif
