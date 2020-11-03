#include <Renderer/InsertionSortUtil.h>

#include <array>
#include <stdexcept>
#include <iostream>

bool InsertionSortUtil::ValueAndIndex::operator<(const ValueAndIndex& other) const {
    return value < other.value;
}

VkDescriptorSetLayout InsertionSortUtil::createDescriptorSetLayout(VkDevice logicalDevice) {
    VkDescriptorSetLayout descriptorSetLayout;

    VkDescriptorSetLayoutBinding valueAndIndexDescriptor = {};
    valueAndIndexDescriptor.binding = 0;
    valueAndIndexDescriptor.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    valueAndIndexDescriptor.descriptorCount = 1;
    valueAndIndexDescriptor.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding wasSwappedDescriptor = {};
    wasSwappedDescriptor.binding = 1;
    wasSwappedDescriptor.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wasSwappedDescriptor.descriptorCount = 1;
    wasSwappedDescriptor.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding infoDescriptor = {};
    infoDescriptor.binding = 2;
    infoDescriptor.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    infoDescriptor.descriptorCount = 1;
    infoDescriptor.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    std::array<VkDescriptorSetLayoutBinding, 3> descriptorSetLayoutBindings = {valueAndIndexDescriptor, wasSwappedDescriptor, infoDescriptor};

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = descriptorSetLayoutBindings.size();
    descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings.data();

    if (vkCreateDescriptorSetLayout(logicalDevice, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute descriptor set layout");
    }

    return descriptorSetLayout;
}

VkDescriptorPool InsertionSortUtil::createDescriptorPool(VkDevice logicalDevice, size_t size) {
    VkDescriptorPool descriptorPool;

    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = size;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = size;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[2].descriptorCount = size;

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets = size;
    descriptorPoolCreateInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    descriptorPoolCreateInfo.pPoolSizes = poolSizes.data();

    if (vkCreateDescriptorPool(logicalDevice, &descriptorPoolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute descriptor pool");
    }

    return descriptorPool;
}

VkDescriptorSet InsertionSortUtil::createDescriptorSet(
    VkDevice logicalDevice,
    VkDescriptorSetLayout& descriptorSetLayout,
    VkDescriptorPool& descriptorPool,
    VkBuffer valueAndIndexBuffer,
    VkBuffer wasSwappedBuffer,
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
    valueAndIndexBufferDescriptor.range = numberOfElements * sizeof(InsertionSortUtil::ValueAndIndex);

    VkWriteDescriptorSet valueAndIndexWriteDescriptorSet = {};
    valueAndIndexWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    valueAndIndexWriteDescriptorSet.dstSet = descriptorSet;
    valueAndIndexWriteDescriptorSet.dstBinding = 0;
    valueAndIndexWriteDescriptorSet.descriptorCount = 1;
    valueAndIndexWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    valueAndIndexWriteDescriptorSet.pBufferInfo = &valueAndIndexBufferDescriptor;

    VkDescriptorBufferInfo wasSwappedBufferDescriptor = {};
    wasSwappedBufferDescriptor.buffer = wasSwappedBuffer;
    wasSwappedBufferDescriptor.offset = 0;
    wasSwappedBufferDescriptor.range = sizeof(uint32_t);

    VkWriteDescriptorSet wasSwappedWriteDescriptorSet = {};
    wasSwappedWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wasSwappedWriteDescriptorSet.dstSet = descriptorSet;
    wasSwappedWriteDescriptorSet.dstBinding = 1;
    wasSwappedWriteDescriptorSet.descriptorCount = 1;
    wasSwappedWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wasSwappedWriteDescriptorSet.pBufferInfo = &wasSwappedBufferDescriptor;

    VkDescriptorBufferInfo infoBufferDescriptor = {};
    infoBufferDescriptor.buffer = infoBuffer;
    infoBufferDescriptor.offset = 0;
    infoBufferDescriptor.range = sizeof(InsertionSortUtil::Info);

    VkWriteDescriptorSet infoWriteDescriptorSet = {};
    infoWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    infoWriteDescriptorSet.dstSet = descriptorSet;
    infoWriteDescriptorSet.dstBinding = 2;
    infoWriteDescriptorSet.descriptorCount = 1;
    infoWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    infoWriteDescriptorSet.pBufferInfo = &infoBufferDescriptor;

    std::array<VkWriteDescriptorSet, 3> writeDescriptorSets = {valueAndIndexWriteDescriptorSet, wasSwappedWriteDescriptorSet, infoWriteDescriptorSet};

    vkUpdateDescriptorSets(logicalDevice, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, nullptr);

    return descriptorSet;
}

VkPipelineLayout InsertionSortUtil::createPipelineLayout(VkDevice logicalDevice, VkDescriptorSetLayout descriptorSetLayout) {
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

VkPipeline InsertionSortUtil::createPipeline(
    VkDevice logicalDevice,
    VkShaderModule shaderModule,
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

VkCommandBuffer InsertionSortUtil::createCommandBuffer(
    VkDevice logicalDevice,
    VkCommandPool commandPool,
    VkPipeline pipeline,
    VkPipelineLayout pipelineLayout,
    VkDescriptorSet descriptorSetOne,
    VkDescriptorSet descriptorSetTwo,
    VkBuffer valueAndIndexBuffer,
    VkBuffer wasSwappedBuffer,
    VkBuffer wasSwappedBufferHostVisible,
    const std::vector<VkBuffer>& steps,
    size_t numberOfElements) {

    VkCommandBuffer commandBuffer;

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(logicalDevice, &commandBufferAllocateInfo, &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute command buffer");
    }

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin compute command buffer");
    }

    VkAccessFlags all =
        VK_ACCESS_UNIFORM_READ_BIT |
        VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT |
        VK_ACCESS_MEMORY_READ_BIT |
        VK_ACCESS_MEMORY_WRITE_BIT ;

    VkMemoryBarrier globalBarrier{};
    globalBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    globalBarrier.srcAccessMask = all;
    globalBarrier.dstAccessMask = all;

    VkBufferMemoryBarrier bufferBarrier = {};
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.srcAccessMask = all;
    bufferBarrier.dstAccessMask = all;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = valueAndIndexBuffer;
    bufferBarrier.offset = 0;
    bufferBarrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &globalBarrier,
        1,
        &bufferBarrier,
        0,
        nullptr);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = sizeof(uint32_t);
    vkCmdCopyBuffer(commandBuffer, wasSwappedBufferHostVisible, wasSwappedBuffer, 1, &copyRegion);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    size_t xGroups = ceil(((float) numberOfElements) / ((float) 2 * X_DIM));
    std::cout << "Number of X groups = " << xGroups << "\n";

    for (int i = 0; i < (xGroups); ++i) {

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            &globalBarrier,
            1,
            &bufferBarrier,
            0,
            nullptr);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSetOne, 0, nullptr);
        vkCmdDispatch(commandBuffer, xGroups, 1, 1);

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            &globalBarrier,
            1,
            &bufferBarrier,
            0,
            nullptr);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSetTwo, 0, nullptr);
        vkCmdDispatch(commandBuffer, xGroups, 1, 1);

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            &globalBarrier,
            1,
            &bufferBarrier,
            0,
            nullptr);

        /*
        VkBufferCopy stepRegion{};
        stepRegion.srcOffset = 0;
        stepRegion.dstOffset = 0;
        stepRegion.size = numberOfElements * sizeof(InsertionSortUtil::ValueAndIndex);

        vkCmdCopyBuffer(
            commandBuffer,
            valueAndIndexBuffer,
            steps[i],
            1,
            &stepRegion);
        */
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &globalBarrier,
        1,
        &bufferBarrier,
        0,
        nullptr);

    vkCmdCopyBuffer(commandBuffer, wasSwappedBuffer, wasSwappedBufferHostVisible, 1, &copyRegion);

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &globalBarrier,
        1,
        &bufferBarrier,
        0,
        nullptr);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end compute command buffer");
    }

    return commandBuffer;
}
