#include <Simulator/Reproducer.h>

#include <Simulator/Agent.h>
#include <Utils/Compute.h>

#include <vector>

namespace {

    constexpr size_t xDim = 512;
    constexpr size_t kNumberOfBindings = 3;

} // namespace anonymous

Reproducer::Reproducer(
    VkDevice logicalDevice,
    VkBuffer agentsInBuffer,
    VkBuffer addressesBuffer,
    VkBuffer agentsOutBuffer,
    uint32_t numberOfElements) {

    m_logicalDevice = logicalDevice;

    m_descriptorSetLayout = Compute::createDescriptorSetLayout(m_logicalDevice, kNumberOfBindings);
    m_descriptorPool = Compute::createDescriptorPool(m_logicalDevice, kNumberOfBindings, 1);

    m_pipelineLayout = Compute::createPipelineLayoutWithPushConstant(m_logicalDevice, m_descriptorSetLayout, sizeof(uint32_t));
    m_pipeline = Compute::createPipeline("src/GLSL/spv/Reproduction.spv", m_logicalDevice, m_pipelineLayout);

    std::vector<Compute::BufferAndSize> bufferAndSizes = {
        {agentsInBuffer, numberOfElements * sizeof(Agent)},
        {addressesBuffer, numberOfElements * sizeof(uint32_t)},
        {agentsOutBuffer, numberOfElements * sizeof(Agent)}
    };

    m_descriptorSet = Compute::createDescriptorSet(
        m_logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        bufferAndSizes);
}

Reproducer::~Reproducer() {
    vkDestroyDescriptorSetLayout(m_logicalDevice, m_descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_logicalDevice, m_descriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_pipeline, nullptr);
}

void Reproducer::recordCommand(VkCommandBuffer commandBuffer, uint32_t numberOfElements) {
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &numberOfElements);

    uint32_t xGroups = ceil(((float) numberOfElements) / ((float) xDim));
    vkCmdDispatch(commandBuffer, xGroups, 1, 1);
}
