#ifndef INSERTION_SORT_H
#define INSERTION_SORT_H

#include <Simulator/InsertionSortUtil.h>

#include <vulkan/vulkan.h>

#include <vector>

class InsertionSort {

private:

    std::vector<InsertionSortUtil::ValueAndIndex> m_serialData;

    VkPhysicalDevice m_physicalDevice;
    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    VkBuffer m_wasSwappedBuffer;
    VkBuffer m_wasSwappedBufferHostVisible;
    VkBuffer m_infoOneBuffer;
    VkBuffer m_infoTwoBuffer;

    VkDeviceMemory m_valueAndIndexBufferMemory;
    VkDeviceMemory m_wasSwappedBufferMemory;
    VkDeviceMemory m_wasSwappedBufferMemoryHostVisible;
    VkDeviceMemory m_infoOneBufferMemory;
    VkDeviceMemory m_infoTwoBufferMemory;

    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipeline;

    VkDescriptorSet m_descriptorSetOne;
    VkDescriptorSet m_descriptorSetTwo;

    VkCommandBuffer m_commandBuffer;

    VkSemaphore m_semaphore;
    VkFence m_fence;

    void setWasSwappedToZero();
    void runSortCommands();
    uint32_t needsSorting();

    void runHelper();

    void printResults();

public:

    VkBuffer m_valueAndIndexBuffer;

    InsertionSort(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool);

    virtual ~InsertionSort() = default;

    void run();

    void cleanUp(VkDevice logicalDevice, VkCommandPool commandPool);
};

#endif
