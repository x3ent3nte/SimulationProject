#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <Renderer/Connector.h>

#include <vulkan/vulkan.h>

#include <memory>
#include <vector>
#include <atomic>

class Simulator {

private:

    std::thread m_simulateTask;
    std::atomic<bool> m_isActive;
    std::shared_ptr<Connector> m_connector;

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

    void simulateNextStep(VkDevice logicalDevice);
    void runSimulatorTask(VkDevice logicalDevice);

public:

    Simulator(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, std::shared_ptr<Connector> connector);

    virtual ~Simulator() = default;

    void simulate(VkDevice logicalDevice);

    void cleanUp(VkDevice logicalDevice);
};

#endif
