#ifndef GRAVITY_H
#define GRAVITY_H

#include <Simulator/Scanner.h>

#include <vulkan/vulkan.h>

#include <memory>
#include <cstdint>

class Gravity {

public:

    Gravity(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        VkBuffer agentsBuffer,
        uint32_t maxNumberOfElements);

    virtual ~Gravity();

    void run(float timeDelta, uint32_t numberOfElements);

private:

    void setNumberOfElements(uint32_t numberOfElements);
    void createCommandBuffer();
    void createCommandBufferIfNecessary(uint32_t numberOfElements);


    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    uint32_t m_currentNumberOfElements;

    VkBuffer m_massiveAgentsBuffer;
    VkDeviceMemory m_massiveAgentsDeviceMemory;

    VkBuffer m_numberOfElementsBuffer;
    VkDeviceMemory m_numberOfElementsDeviceMemory;

    VkBuffer m_numberOfElementsHostVisibleBuffer;
    VkDeviceMemory m_numberOfElementsHostVisibleDeviceMemory;

    VkBuffer m_timeDeltaBuffer;
    VkDeviceMemory m_timeDeltaDeviceMemory;

    VkBuffer m_timeDeltaHostVisibleBuffer;
    VkDeviceMemory m_timeDeltaHostVisibleDeviceMemory;

    VkDescriptorSetLayout m_mapDescriptorSetLayout;
    VkDescriptorPool m_mapDescriptorPool;
    VkPipelineLayout m_mapPipelineLayout;
    VkPipeline m_mapPipeline;
    VkDescriptorSet m_mapDescriptorSet;

    VkDescriptorSetLayout m_scatterDescriptorSetLayout;
    VkDescriptorPool m_scatterDescriptorPool;
    VkPipelineLayout m_scatterPipelineLayout;
    VkPipeline m_scatterPipeline;
    VkDescriptorSet m_scatterDescriptorSet;

    VkDescriptorSetLayout m_gravityDescriptorSetLayout;
    VkDescriptorPool m_gravityDescriptorPool;
    VkPipelineLayout m_gravityPipelineLayout;
    VkPipeline m_gravityPipeline;
    VkDescriptorSet m_gravityDescriptorSet;

    VkFence m_fence;

    std::shared_ptr<Scanner<int32_t>> m_scanner;

    VkCommandBuffer m_setNumberOfElementsCommandBuffer;
    VkCommandBuffer m_commandBuffer;
};

#endif
