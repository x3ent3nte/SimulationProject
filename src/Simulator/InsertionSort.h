#ifndef INSERTION_SORT_H
#define INSERTION_SORT_H

#include <Simulator/InsertionSortUtil.h>

#include <vulkan/vulkan.h>

#include <vector>

class InsertionSort {

private:

    VkPhysicalDevice m_physicalDevice;
    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    VkBuffer m_wasSwappedBuffer;
    VkBuffer m_wasSwappedBufferHostVisible;
    VkBuffer m_dataSizeBuffer;
    VkBuffer m_dataSizeBufferHostVisible;
    VkBuffer m_offsetOneBuffer;
    VkBuffer m_offsetTwoBuffer;

    VkDeviceMemory m_valueAndIndexBufferMemory;
    VkDeviceMemory m_wasSwappedBufferMemory;
    VkDeviceMemory m_wasSwappedBufferMemoryHostVisible;
    VkDeviceMemory m_dataSizeBufferMemory;
    VkDeviceMemory m_dataSizeBufferMemoryHostVisible;
    VkDeviceMemory m_offsetOneBufferMemory;
    VkDeviceMemory m_offsetTwoBufferMemory;

    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipeline;

    VkDescriptorSet m_descriptorSetOne;
    VkDescriptorSet m_descriptorSetTwo;

    VkCommandBuffer m_commandBuffer;
    VkCommandBuffer m_setDataSizeCommandBuffer;

    VkSemaphore m_semaphore;
    VkFence m_fence;

    void setDataSize(uint32_t dataSize);
    void createCommandBuffer(uint32_t dataSize);

    void setWasSwappedToZero();
    void runSortCommands();
    uint32_t needsSorting();

    uint32_t m_currentDataSize;

public:

    VkBuffer m_valueAndIndexBuffer;

    InsertionSort(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        uint32_t numberOfElements);

    virtual ~InsertionSort();

    void run(uint32_t dataSize);
};

#endif
