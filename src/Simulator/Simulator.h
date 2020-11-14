#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <Renderer/Connector.h>
#include <Simulator/InsertionSort.h>

#include <vulkan/vulkan.h>

#include <memory>
#include <vector>
#include <atomic>

class Simulator {

private:

    std::thread m_simulateTask;
    std::atomic<bool> m_isActive;
    std::shared_ptr<Connector> m_connector;

    VkDevice m_logicalDevice;

    VkQueue m_computeQueue;
    VkFence m_computeFence;

    VkDescriptorSetLayout m_computeDescriptorSetLayout;

    std::vector<VkDescriptorPool> m_computeDescriptorPools;
    std::vector<VkDescriptorSet> m_computeDescriptorSets;

    VkCommandPool m_computeCommandPool;

    std::vector<VkPipeline> m_computePipelines;
    std::vector<VkPipelineLayout> m_computePipelineLayouts;
    std::vector<VkCommandBuffer> m_computeCommandBuffers;

    VkBuffer m_agentsBuffer;
    VkDeviceMemory m_agentsBufferMemory;

    std::shared_ptr<InsertionSort> m_insertionSort;

    void simulateNextStep(VkCommandBuffer commandBuffer);
    void runSimulatorTask();

public:

    Simulator(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue computeQueue,
        VkCommandPool computeCommandPool,
        std::shared_ptr<Connector> connector);

    virtual ~Simulator();

    void simulate();

    void stopSimulation(VkPhysicalDevice physicalDevice);
};

#endif
