#ifndef SIMULATION_STATE_WRITER_H
#define SIMULATION_STATE_WRITER_H

#include <Simulator/Agent.h>

#include <vulkan/vulkan.h>

#include <vector>

class SimulationStateWriterFunction {

private:

public:

};

class SimulationStateWriter {

private:

    VkDevice m_logicalDevice;

    VkQueue m_queue;
    VkCommandPool m_commandPool;

    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;

    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipeline;

public:

    SimulationStateWriter(
        VkDevice logicalDevice,
        size_t descriptorPoolSize);

    virtual ~SimulationStateWriter();
};

#endif
