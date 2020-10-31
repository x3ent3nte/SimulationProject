#include <Renderer/InsertionSort.h>

#include <array>
#include <stdexcept>

VkDescriptorSetLayout InsertionSort::createDescriptorSetLayout(VkDevice logicalDevice) {
    VkDescriptorSetLayout descriptorSetLayout;

    VkDescriptorSetLayoutBinding valueAndIndexDescriptor = {};
    valueAndIndexDescriptor.binding = 0;
    valueAndIndexDescriptor.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    valueAndIndexDescriptor.descriptorCount = 1;
    valueAndIndexDescriptor.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding infoDescriptor = {};
    infoDescriptor.binding = 1;
    infoDescriptor.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    infoDescriptor.descriptorCount = 1;
    infoDescriptor.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    std::array<VkDescriptorSetLayoutBinding, 2> descriptorSetLayoutBindings = {valueAndIndexDescriptor, infoDescriptor};

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = 2;
    descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings.data();

    if (vkCreateDescriptorSetLayout(logicalDevice, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute descriptor set layout");
    }

    return descriptorSetLayout;
}

VkDescriptorPool createDescriptorPool(VkDevice logicalDevice, size_t maxSets) {
    VkDescriptorPool descriptorPool;

    VkDescriptorPoolSize descriptorPoolSize = {};
    descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorPoolSize.descriptorCount = 2;

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets = maxSets;
    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;

    if (vkCreateDescriptorPool(logicalDevice, &descriptorPoolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute descriptor pool");
    }

    return descriptorPool;
}

VkDescriptorSet InsertionSort::createDescriptorSet(
    VkDevice logicalDevice,
    VkDescriptorSetLayout& descriptorSetLayout,
    VkDescriptorPool& descriptorPool,
    VkBuffer valueAndIndexBuffer,
    VkBuffer infoBuffer,
    size_t numberOfElements) {

    VkDescriptorSet descriptorSet;

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

    if (vkAllocateDescriptorSets(logicalDevice, &descriptorSetAllocateInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute descriptor sets");
    }

    VkDescriptorBufferInfo valueAndIndexBufferDescriptor = {};
    valueAndIndexBufferDescriptor.buffer = valueAndIndexBuffer;
    valueAndIndexBufferDescriptor.offset = 0;
    valueAndIndexBufferDescriptor.range = numberOfElements * sizeof(InsertionSort::ValueAndIndex);

    VkWriteDescriptorSet valueAndIndexWriteDescriptorSet = {};
    valueAndIndexWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    valueAndIndexWriteDescriptorSet.dstSet = descriptorSet;
    valueAndIndexWriteDescriptorSet.dstBinding = 0;
    valueAndIndexWriteDescriptorSet.descriptorCount = 1;
    valueAndIndexWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    valueAndIndexWriteDescriptorSet.pBufferInfo = &valueAndIndexBufferDescriptor;

    VkDescriptorBufferInfo infoBufferDescriptor = {};
    infoBufferDescriptor.buffer = infoBuffer;
    infoBufferDescriptor.offset = 0;
    infoBufferDescriptor.range = sizeof(InsertionSort::Info);

    VkWriteDescriptorSet infoWriteDescriptorSet = {};
    infoWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    infoWriteDescriptorSet.dstSet = descriptorSet;
    infoWriteDescriptorSet.dstBinding = 1;
    infoWriteDescriptorSet.descriptorCount = 1;
    infoWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    infoWriteDescriptorSet.pBufferInfo = &infoBufferDescriptor;

    std::array<VkWriteDescriptorSet, 2> writeDescriptorSets = {valueAndIndexWriteDescriptorSet, infoWriteDescriptorSet};

    vkUpdateDescriptorSets(logicalDevice, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, nullptr);

    return descriptorSet;
}

VkPipelineLayout InsertionSort::createPipelineLayout(VkDevice logicalDevice, VkDescriptorSetLayout descriptorSetLayout) {
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

VkPipeline InsertionSort::createPipeline(
    VkDevice logicalDevice,
    VkShaderModule shaderModule,
    VkDescriptorSetLayout descriptorSetLayout,
    VkPipelineLayout pipelineLayout) {

    VkPipeline pipeline;

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

    return pipeline;
}
