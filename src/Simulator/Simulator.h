#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <Renderer/Connector.h>
#include <Renderer/Mesh.h>
#include <Simulator/InsertionSorter.h>
#include <Simulator/Scanner.h>
#include <Simulator/Collider.h>
#include <Simulator/Gravity.h>
#include <Simulator/Boids.h>
#include <Simulator/Scanner.h>
#include <Simulator/InputTerminal.h>
#include <Simulator/SimulationStateWriter.h>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <vector>
#include <atomic>

class Simulator {

private:

    std::thread m_simulateTask;
    std::atomic<bool> m_isActive;
    std::shared_ptr<Connector> m_connector;
    std::shared_ptr<InputTerminal> m_inputTerminal;

    std::shared_ptr<Mesh> m_mesh;

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

    VkBuffer m_timeDeltaBuffer;
    VkDeviceMemory m_timeDeltaDeviceMemory;

    VkBuffer m_timeDeltaBufferHostVisible;
    VkDeviceMemory m_timeDeltaDeviceMemoryHostVisible;

    VkBuffer m_numberOfElementsBuffer;
    VkDeviceMemory m_numberOfElementsDeviceMemory;

    VkBuffer m_numberOfElementsBufferHostVisible;
    VkDeviceMemory m_numberOfElementsDeviceMemoryHostVisible;

    VkBuffer m_playerRenderInfosBuffer;
    VkDeviceMemory m_playerRenderInfosDeviceMemory;

    VkBuffer m_playerRenderInfosHostVisibleBuffer;
    VkDeviceMemory m_playerRenderInfosHostVisibleDeviceMemory;

    uint32_t m_currentNumberOfElements;

    std::shared_ptr<Collider> m_collider;
    std::shared_ptr<AgentSorter> m_agentSorter;
    std::shared_ptr<Gravity> m_gravity;
    std::shared_ptr<Boids> m_boids;
    std::vector<std::shared_ptr<SimulationStateWriterFunction>> m_simulationStateWriterFunctions;

    void simulateNextStep(VkCommandBuffer commandBuffer, float timeDelta);
    void runSimulatorTask();
    void runSimulatorStateWriterFunction(uint32_t numberOfPlayers);

public:

    Simulator(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue computeQueue,
        VkCommandPool computeCommandPool,
        std::shared_ptr<Connector> connector,
        std::shared_ptr<InputTerminal> inputTerminal,
        std::shared_ptr<Mesh> mesh,
        uint32_t maxNumberOfAgents,
        uint32_t maxNumberOfPlayers);

    virtual ~Simulator();

    void simulate();

    void stopSimulation(VkPhysicalDevice physicalDevice);
};

#endif
