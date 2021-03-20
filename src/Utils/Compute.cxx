#include <Utils/Compute.h>

#include <Utils/Utils.h>

#include <stdexcept>

VkDescriptorSetLayout Compute::createDescriptorSetLayout(VkDevice logicalDevice, size_t numberOfBuffers) {

    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings(numberOfBuffers);

    for (int i = 0; i < numberOfBuffers; ++i) {
        descriptorSetLayoutBindings[i].binding = i;
        descriptorSetLayoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindings[i].descriptorCount = 1;
        descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = descriptorSetLayoutBindings.size();
    descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings.data();

    VkDescriptorSetLayout descriptorSetLayout;
    if (vkCreateDescriptorSetLayout(logicalDevice, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute descriptor set layout");
    }
    return descriptorSetLayout;
}

VkDescriptorPool Compute::createDescriptorPool(VkDevice logicalDevice, size_t numberOfBindings, size_t maxSets) {
    VkDescriptorPool descriptorPool;

    std::vector<VkDescriptorPoolSize> poolSizes(numberOfBindings);

    for (int i = 0; i < numberOfBindings; ++i) {
        poolSizes[i].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[i].descriptorCount = maxSets;
    }

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets = maxSets;
    descriptorPoolCreateInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    descriptorPoolCreateInfo.pPoolSizes = poolSizes.data();

    if (vkCreateDescriptorPool(logicalDevice, &descriptorPoolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute descriptor pool");
    }

    return descriptorPool;
}

VkDescriptorSet Compute::createDescriptorSet(
    VkDevice logicalDevice,
    VkDescriptorSetLayout& descriptorSetLayout,
    VkDescriptorPool& descriptorPool,
    const std::vector<Compute::BufferAndSize>& bufferAndSizes) {

    VkDescriptorSet descriptorSet;

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

    if (vkAllocateDescriptorSets(logicalDevice, &descriptorSetAllocateInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute descriptor sets");
    }

    std::vector<VkDescriptorBufferInfo> bufferDescriptors(bufferAndSizes.size());
    std::vector<VkWriteDescriptorSet> descriptors(bufferAndSizes.size());

    for (int i = 0; i < bufferAndSizes.size(); ++i) {
        bufferDescriptors[i].buffer = bufferAndSizes[i].buffer;
        bufferDescriptors[i].offset = 0;
        bufferDescriptors[i].range = bufferAndSizes[i].size;

        descriptors[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptors[i].dstSet = descriptorSet;
        descriptors[i].dstBinding = i;
        descriptors[i].descriptorCount = 1;
        descriptors[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptors[i].pBufferInfo = &bufferDescriptors[i];
    }

    vkUpdateDescriptorSets(logicalDevice, descriptors.size(), descriptors.data(), 0, nullptr);
    return descriptorSet;
}

VkPipelineLayout Compute::createPipelineLayout(VkDevice logicalDevice, VkDescriptorSetLayout descriptorSetLayout) {
    VkPipelineLayout pipelineLayout;

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;

    if (vkCreatePipelineLayout(logicalDevice, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    return pipelineLayout;
}

VkPipelineLayout Compute::createPipelineLayoutWithPushConstant(
    VkDevice logicalDevice,
    VkDescriptorSetLayout descriptorSetLayout,
    uint32_t size) {

    VkPushConstantRange pushConstant = {};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = size;

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstant;

    VkPipelineLayout pipelineLayout;
    if (vkCreatePipelineLayout(logicalDevice, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    return pipelineLayout;
}

VkPipeline Compute::createPipeline(
    const std::string& shaderPath,
    VkDevice logicalDevice,
    VkPipelineLayout pipelineLayout) {

    VkPipeline pipeline;

    auto shaderCode = Utils::readFile(shaderPath);
    VkShaderModule shaderModule = Utils::createShaderModule(logicalDevice, shaderCode);

    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = shaderModule;
    shaderStageCreateInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage = shaderStageCreateInfo;
    pipelineCreateInfo.layout = pipelineLayout;

    if (vkCreateComputePipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline");
    }

    vkDestroyShaderModule(logicalDevice, shaderModule, nullptr);

    return pipeline;
}
