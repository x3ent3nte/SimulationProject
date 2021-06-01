#ifndef SHADER_FUNCTION_H
#define SHADER_FUNCTION_H

#include <Utils/Compute.h>

#include <vulkan/vulkan.h>

#include <memory>
#include <string>

class ShaderFunction {

public:

    ShaderFunction(
        VkDevice logicalDevice,
        size_t numberOfBindings,
        const std::string& shaderPath);

    virtual ~ShaderFunction();

    VkDevice m_logicalDevice;

    VkDescriptorSetLayout m_descriptorSetLayout;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipeline;

    const size_t m_numberOfBindings;

private:

};


class ShaderPool {

public:

    ShaderPool(
        std::shared_ptr<ShaderFunction> shaderFn,
        size_t maxSets);

    virtual ~ShaderPool();

    std::shared_ptr<ShaderFunction> m_shaderFn;

    VkDescriptorPool m_descriptorPool;

private:

};

class ShaderLambda {

public:

    ShaderLambda(
        std::shared_ptr<ShaderPool> shaderPool,
        const std::vector<Compute::BufferAndSize>& bufferAndSizes);

    virtual ~ShaderLambda();

    std::shared_ptr<ShaderPool> m_shaderPool;
    VkDescriptorSet m_descriptorSet;

    void bind(VkCommandBuffer commandBuffer, size_t x, size_t y, size_t z);

private:

};

#endif
