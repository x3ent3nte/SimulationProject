#ifndef INSERTION_SORT_H
#define INSERTION_SORT_H

#include <Renderer/InsertionSortUtil.h>

#include <vulkan/vulkan.h>

class InsertionSort {

private:

    VkBuffer m_valueAndIndexBuffer;
    VkBuffer m_wasSwappedBuffer;
    VkBuffer m_infoOneBuffer;
    VkBuffer m_infoTwoBuffer;

    VkDeviceMemory m_valueAndIndexBufferMemory;
    VkDeviceMemory m_wasSwappedBufferMemory;
    VkDeviceMemory m_infoOneBufferMemory;
    VkDeviceMemory m_infoTwoBufferMemory;

    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipeline;

    VkDescriptorSet m_descriptorSetOne;
    VkCommandBuffer m_commandBufferOne;

    VkDescriptorSet m_descriptorSetTwo;
    VkCommandBuffer m_commandBufferTwo;

    VkSemaphore m_semaphore;
    VkFence m_fence;

public:

    InsertionSort(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool);

    virtual ~InsertionSort() = default;

    void run();

    void cleanUp(VkDevice logicalDevice, VkCommandPool commandPool);
};

#endif
