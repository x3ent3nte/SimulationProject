#include <Utils/ShaderFunction.h>

ShaderFunction::ShaderFunction(
    VkDevice logicalDevice,
    size_t numberOfBindings,
    const std::string& shaderPath)
    : m_logicalDevice(logicalDevice)
    , m_numberOfBindings(numberOfBindings) {

    m_descriptorSetLayout = Compute::createDescriptorSetLayout(m_logicalDevice, numberOfBindings);;
    m_pipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_descriptorSetLayout);
    m_pipeline = Compute::createPipeline(shaderPath, m_logicalDevice, m_pipelineLayout);
}

ShaderFunction::~ShaderFunction() {
    vkDestroyDescriptorSetLayout(m_logicalDevice, m_descriptorSetLayout, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_pipeline, nullptr);
}


ShaderPool::ShaderPool(
    std::shared_ptr<ShaderFunction> shaderFn,
    size_t maxSets)
    : m_shaderFn(shaderFn) {

    m_descriptorPool = Compute::createDescriptorPool(m_shaderFn->m_logicalDevice, m_shaderFn->m_numberOfBindings, maxSets);
}

ShaderPool::~ShaderPool() {
    vkDestroyDescriptorPool(m_shaderFn->m_logicalDevice, m_descriptorPool, nullptr);
}


ShaderLambda::ShaderLambda(
    std::shared_ptr<ShaderPool> shaderPool,
    const std::vector<Compute::BufferAndSize>& bufferAndSizes)
    : m_shaderPool(shaderPool) {

     m_descriptorSet = Compute::createDescriptorSet(
        m_shaderPool->m_shaderFn->m_logicalDevice,
        m_shaderPool->m_shaderFn->m_descriptorSetLayout,
        m_shaderPool->m_descriptorPool,
        bufferAndSizes);
}

ShaderLambda::~ShaderLambda() {

}

void ShaderLambda::bind(VkCommandBuffer commandBuffer, size_t x, size_t y, size_t z) {
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_shaderPool->m_shaderFn->m_pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_shaderPool->m_shaderFn->m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
    vkCmdDispatch(commandBuffer, x, y, z);
}

