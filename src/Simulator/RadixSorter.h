#ifndef RADIX_SORTER_H
#define RADIX_SORTER_H

#include <Simulator/Scanner.h>
#include <Renderer/MyGLM.h>

#include <vulkan/vulkan.h>

#include <memory>

class RadixSorter {

public:

    struct ValueAndIndex {
        uint32_t value;
        uint32_t index;
    };

    RadixSorter(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        uint32_t maxNumberOfElements);

    virtual ~RadixSorter();

    void run(uint32_t numberOfElements);

    VkBuffer m_dataBuffer;

private:

    void runCommandAndWaitForFence(VkCommandBuffer commandBuffer);
    void setNumberOfElements(uint32_t numberOfElements);
    void setRadix(uint32_t radix);
    void destroyCommandBuffers();
    void createCommandBuffers();
    void createCommandBuffersIfNecessary(uint32_t numberOfElements);
    bool needsSorting();
    void sort();

    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    uint32_t m_currentNumberOfElements;

    // Pipeline
    VkDescriptorSetLayout m_mapDescriptorSetLayout;
    VkDescriptorPool m_mapDescriptorPool;
    VkPipelineLayout m_mapPipelineLayout;
    VkPipeline m_mapPipeline;

    VkDescriptorSet m_mapDescriptorSetOne;
    VkDescriptorSet m_mapDescriptorSetTwo;

    VkDescriptorSetLayout m_scatterDescriptorSetLayout;
    VkDescriptorPool m_scatterDescriptorPool;
    VkPipelineLayout m_scatterPipelineLayout;
    VkPipeline m_scatterPipeline;

    VkDescriptorSet m_scatterDescriptorSetOne;
    VkDescriptorSet m_scatterDescriptorSetTwo;

    // Buffers
    VkDeviceMemory m_dataDeviceMemory;

    VkBuffer m_otherBuffer;
    VkDeviceMemory m_otherDeviceMemory;

    VkBuffer m_radixBuffer;
    VkDeviceMemory m_radixDeviceMemory;

    VkBuffer m_radixHostVisibleBuffer;
    VkDeviceMemory m_radixHostVisibleDeviceMemory;

    VkBuffer m_numberOfElementsBuffer;
    VkDeviceMemory m_numberOfElementsDeviceMemory;

    VkBuffer m_numberOfElementsHostVisibleBuffer;
    VkDeviceMemory m_numberOfElementsHostVisibleDeviceMemory;

    // Commands
    VkCommandBuffer m_setNumberOfElementsCommandBuffer;
    VkCommandBuffer m_copyBuffersCommandBuffer;

    VkCommandBuffer m_commandBufferOne;
    VkCommandBuffer m_commandBufferTwo;

    VkFence m_fence;

    std::shared_ptr<Scanner<glm::uvec4>> m_scanner;
};

#endif
