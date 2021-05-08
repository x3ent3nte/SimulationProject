#ifndef SIMULATION_STATE_WRITER_H
#define SIMULATION_STATE_WRITER_H

#include <vulkan/vulkan.h>

#include <vector>
#include <memory>

class SimulationStateWriter {

public:

    VkDevice m_logicalDevice;

    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;

    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipeline;

    SimulationStateWriter(
        VkDevice logicalDevice,
        size_t descriptorPoolSize);

    virtual ~SimulationStateWriter();
};

class SimulationStateWriterFunction {

private:

    const std::shared_ptr<SimulationStateWriter> m_simulationStateWriter;

    VkDescriptorSet m_descriptorSet;

public:

    SimulationStateWriterFunction::SimulationStateWriterFunction(
        std::shared_ptr<SimulationStateWriter> simulationStateWriter,
        VkBuffer agentsBuffer,
        VkBuffer agentRenderInfosBuffer,
        VkBuffer playerRenderInfosBuffer,
        uint32_t maxNumberOfAgents,
        uint32_t maxNumberOfPlayers);

    virtual ~SimulationStateWriterFunction();

    void recordCommand(VkCommandBuffer commandBuffer, uint32_t numberOfAgents);
};

#endif
