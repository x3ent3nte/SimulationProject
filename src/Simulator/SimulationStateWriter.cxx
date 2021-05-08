#include <Simulator/SimulationStateWriter.h>

#include <Simulator/Agent.h>

#include <Utils/Compute.h>

namespace {

    constexpr size_t xDim = 512;
    constexpr size_t kNumberOfBindings = 3;

} // namespace anonymous

SimulationStateWriter::SimulationStateWriter(
    VkDevice logicalDevice,
    size_t descriptorPoolSize) {

    m_logicalDevice = logicalDevice;

    m_descriptorSetLayout = Compute::createDescriptorSetLayout(m_logicalDevice, kNumberOfBindings);
    m_descriptorPool = Compute::createDescriptorPool(m_logicalDevice, kNumberOfBindings, descriptorPoolSize);

    m_pipelineLayout = Compute::createPipelineLayoutWithPushConstant(m_logicalDevice, m_descriptorSetLayout, sizeof(uint32_t));
    m_pipeline = Compute::createPipeline("src/GLSL/spv/SimulationOutput.spv", m_logicalDevice, m_pipelineLayout);
}

SimulationStateWriter::~SimulationStateWriter() {
    vkDestroyDescriptorSetLayout(m_logicalDevice, m_descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_logicalDevice, m_descriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_pipeline, nullptr);
}

SimulationStateWriterFunction::SimulationStateWriterFunction(
    std::shared_ptr<SimulationStateWriter> simulationStateWriter,
    VkBuffer agentsBuffer,
    VkBuffer agentRenderInfosBuffer,
    VkBuffer playerRenderInfosBuffer,
    uint32_t maxNumberOfAgents,
    uint32_t maxNumberOfPlayers)
    : m_simulationStateWriter(simulationStateWriter) {

    std::vector<Compute::BufferAndSize> bufferAndSizes = {
        {agentsBuffer, maxNumberOfAgents * sizeof(Agent)},
        {agentRenderInfosBuffer, maxNumberOfAgents * sizeof(AgentRenderInfo)},
        {playerRenderInfosBuffer, maxNumberOfPlayers * sizeof(AgentRenderInfo)}
    };

    m_descriptorSet = Compute::createDescriptorSet(
        m_simulationStateWriter->m_logicalDevice,
        m_simulationStateWriter->m_descriptorSetLayout,
        m_simulationStateWriter->m_descriptorPool,
        bufferAndSizes);
}

SimulationStateWriterFunction::~SimulationStateWriterFunction() {

}

void SimulationStateWriterFunction::recordCommand(VkCommandBuffer commandBuffer, uint32_t numberOfAgents) {
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_simulationStateWriter->m_pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_simulationStateWriter->m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, m_simulationStateWriter->m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &numberOfAgents);

    uint32_t xGroups = ceil(((float) numberOfAgents) / ((float) xDim));
    vkCmdDispatch(commandBuffer, xGroups, 1, 1);
}


