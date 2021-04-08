#include <Simulator/SimulationStateWriter.h>

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


