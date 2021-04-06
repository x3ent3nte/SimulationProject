#ifndef BOIDS_H
#define BOIDS_H

#include <vulkan/vulkan.h>

#include <Simulator/Scanner.h>
#include <Simulator/Reproducer.h>
#include <Renderer/KeyboardControl.h>

#include <memory>
#include <vector>

class Boids {
private:
    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;
    VkBuffer m_agentsBuffer;
    uint32_t m_currentNumberOfElements;
    const uint32_t m_maxNumberOfPlayers;

    VkBuffer m_otherAgentsBuffer;
    VkDeviceMemory m_otherAgentsDeviceMemory;

    VkBuffer m_timeDeltaBuffer;
    VkDeviceMemory m_timeDeltaDeviceMemory;

    VkBuffer m_timeDeltaBufferHostVisible;
    VkDeviceMemory m_timeDeltaDeviceMemoryHostVisible;

    VkBuffer m_numberOfElementsBuffer;
    VkDeviceMemory m_numberOfElementsDeviceMemory;

    VkBuffer m_numberOfElementsBufferHostVisible;
    VkDeviceMemory m_numberOfElementsDeviceMemoryHostVisible;

    VkBuffer m_playerInputStatesBuffer;
    VkDeviceMemory m_playerInputStatesDeviceMemory;

    VkBuffer m_playerInputStatesHostVisibleBuffer;
    VkDeviceMemory m_playerInputStatesHostVisibleDeviceMemory;

    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipeline;
    VkDescriptorSet m_descriptorSet;

    VkCommandBuffer m_setNumberOfElementsCommandBuffer;
    VkCommandBuffer m_commandBuffer;

    VkFence m_fence;

    std::shared_ptr<Scanner> m_scanner;
    std::shared_ptr<Reproducer> m_reproducer;


    void copyPlayerInputStates(std::vector<uint32_t>& playerInputStates);
    void updateNumberOfElementsIfNecessary(uint32_t numberOfElements);
    void createCommandBuffer(uint32_t numberOfElements);

    uint32_t extractNumberOfElements();

public:

    Boids(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        VkBuffer agentsBuffer,
        uint32_t numberOfElements,
        uint32_t maxNumberOfPlayers);

    virtual ~Boids();

    uint32_t run(float timeDelta, uint32_t numberOfElements, std::vector<uint32_t>& playerInputStates);
};

#endif
