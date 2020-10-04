#include <Renderer/Simulator.h>

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES

#include <Renderer/Buffer.h>
#include <Renderer/Utils.h>
#include <Renderer/PhysicalDevice.h>

#include <array>
#include <stdexcept>
#include <iostream>

#define X_DIM 512
#define NUM_ELEMENTS 16 * X_DIM
//#define BUFFER_SIZE NUM_ELEMENTS * sizeof(float)
#define BUFFER_SIZE NUM_ELEMENTS * sizeof(glm::vec2)

namespace {

    VkDescriptorSetLayout createComputeDescriptorSetLayout(VkDevice logicalDevice) {
        VkDescriptorSetLayout descriptorSetLayout;

        VkDescriptorSetLayoutBinding descriptorSetLayoutBindingA = {};
        descriptorSetLayoutBindingA.binding = 0;
        descriptorSetLayoutBindingA.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindingA.descriptorCount = 1;
        descriptorSetLayoutBindingA.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding descriptorSetLayoutBindingB = {};
        descriptorSetLayoutBindingB.binding = 1;
        descriptorSetLayoutBindingB.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindingB.descriptorCount = 1;
        descriptorSetLayoutBindingB.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding descriptorSetLayoutBindingC = {};
        descriptorSetLayoutBindingC.binding = 2;
        descriptorSetLayoutBindingC.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindingC.descriptorCount = 1;
        descriptorSetLayoutBindingC.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        std::array<VkDescriptorSetLayoutBinding, 3> descriptorSetLayoutBindings = {descriptorSetLayoutBindingA, descriptorSetLayoutBindingB, descriptorSetLayoutBindingC};

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.bindingCount = 3;
        descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings.data();

        if (vkCreateDescriptorSetLayout(logicalDevice, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute descriptor set layout");
        }

        return descriptorSetLayout;
    }

    VkDescriptorPool createComputeDescriptorPool(VkDevice logicalDevice) {
        VkDescriptorPool descriptorPool;

        VkDescriptorPoolSize descriptorPoolSize = {};
        descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorPoolSize.descriptorCount = 3;

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
        descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.maxSets = 1;
        descriptorPoolCreateInfo.poolSizeCount = 1;
        descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;

        if (vkCreateDescriptorPool(logicalDevice, &descriptorPoolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute descriptor pool");
        }

        return descriptorPool;
    }

    struct Agent {
        glm::vec2 position;
        glm::vec2 velocity;
    };

    void initializeComputeBuffers(VkDevice logicalDevice, VkDeviceMemory memoryA, VkDeviceMemory memoryB) {
        void* mappedMemoryA = NULL;
        vkMapMemory(logicalDevice, memoryA, 0, NUM_ELEMENTS * sizeof(Agent), 0, & mappedMemoryA);
        Agent* floatMappedMemoryA = (Agent*) mappedMemoryA;
        for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
            floatMappedMemoryA[i] = {glm::vec2(i, i + 1), glm::vec2(i, i + 2)};
        }
        vkUnmapMemory(logicalDevice, memoryA);

        void* mappedMemoryB = NULL;
        vkMapMemory(logicalDevice, memoryB, 0, BUFFER_SIZE, 0, & mappedMemoryB);
        glm::vec2* floatMappedMemoryB = (glm::vec2*) mappedMemoryB;
        for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
            floatMappedMemoryB[i] = glm::vec2(i, (2 * i) + 1);
        }
        vkUnmapMemory(logicalDevice, memoryB);
    }

    VkDescriptorSet createComputeDescriptorSet(
        VkDevice logicalDevice,
        VkDescriptorSetLayout& descriptorSetLayout ,
        VkDescriptorPool& descriptorPool,
        VkBuffer bufferA,
        VkBuffer bufferB,
        VkBuffer bufferC,
        size_t bufferSize) {

        VkDescriptorSet descriptorSet;

        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
        descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocateInfo.descriptorPool = descriptorPool;
        descriptorSetAllocateInfo.descriptorSetCount = 1;
        descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

        if (vkAllocateDescriptorSets(logicalDevice, &descriptorSetAllocateInfo, &descriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute descriptor sets");
        }

        VkDescriptorBufferInfo descriptorBufferInfoA = {};
        descriptorBufferInfoA.buffer = bufferA;
        descriptorBufferInfoA.offset = 0;
        descriptorBufferInfoA.range = NUM_ELEMENTS * sizeof(Agent);

        VkWriteDescriptorSet writeDescriptorSetA = {};
        writeDescriptorSetA.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSetA.dstSet = descriptorSet;
        writeDescriptorSetA.dstBinding = 0;
        writeDescriptorSetA.descriptorCount = 1;
        writeDescriptorSetA.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSetA.pBufferInfo = &descriptorBufferInfoA;

        VkDescriptorBufferInfo descriptorBufferInfoB = {};
        descriptorBufferInfoB.buffer = bufferB;
        descriptorBufferInfoB.offset = 0;
        descriptorBufferInfoB.range = bufferSize;

        VkWriteDescriptorSet writeDescriptorSetB = {};
        writeDescriptorSetB.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSetB.dstSet = descriptorSet;
        writeDescriptorSetB.dstBinding = 1;
        writeDescriptorSetB.descriptorCount = 1;
        writeDescriptorSetB.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSetB.pBufferInfo = &descriptorBufferInfoB;

        VkDescriptorBufferInfo descriptorBufferInfoC = {};
        descriptorBufferInfoC.buffer = bufferC;
        descriptorBufferInfoC.offset = 0;
        descriptorBufferInfoC.range = NUM_ELEMENTS * sizeof(glm::vec3);

        VkWriteDescriptorSet writeDescriptorSetC = {};
        writeDescriptorSetC.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSetC.dstSet = descriptorSet;
        writeDescriptorSetC.dstBinding = 2;
        writeDescriptorSetC.descriptorCount = 1;
        writeDescriptorSetC.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSetC.pBufferInfo = &descriptorBufferInfoC;

        std::array<VkWriteDescriptorSet, 3> writeDescriptorSets = {writeDescriptorSetA, writeDescriptorSetB, writeDescriptorSetC};

        vkUpdateDescriptorSets(logicalDevice, 3, writeDescriptorSets.data(), 0, nullptr);

        return descriptorSet;
    }

    void createComputePipeline(
        VkDevice logicalDevice,
        VkDescriptorSetLayout descriptorSetLayout,
        VkPipelineLayout& pipelineLayout,
        VkPipeline& pipeline) {

        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
        pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.setLayoutCount = 1;
        pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;

        if (vkCreatePipelineLayout(logicalDevice, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout");
        }

        auto kernelShaderCode = Utils::readFile("src/GLSL/kernel.spv");

        VkShaderModule kernelShaderModule = Utils::createShaderModule(logicalDevice, kernelShaderCode);

        VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
        shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStageCreateInfo.module = kernelShaderModule;
        shaderStageCreateInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineCreateInfo = {};
        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.stage = shaderStageCreateInfo;
        pipelineCreateInfo.layout = pipelineLayout;

        if (vkCreateComputePipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute pipeline");
        }

        vkDestroyShaderModule(logicalDevice, kernelShaderModule, nullptr);
    }

    VkCommandPool createComputeCommandPool(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, size_t computeQueueIndex) {
        VkCommandPool commandPool;

        VkCommandPoolCreateInfo commandPoolCreateInfo = {};
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.flags = 0;
        commandPoolCreateInfo.queueFamilyIndex = computeQueueIndex;

        if (vkCreateCommandPool(logicalDevice, &commandPoolCreateInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute command pool");
        }

        return commandPool;
    }

    VkCommandBuffer createComputeCommandBuffer(
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkPipeline pipeline,
        VkPipelineLayout pipelineLayout,
        VkDescriptorSet descriptorSet) {

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
        beginInfo.flags = 0;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin compute command buffer");
        }

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

        size_t xThreads = NUM_ELEMENTS / X_DIM;
        vkCmdDispatch(commandBuffer, xThreads, 1, 1);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to end compute command buffer");
        }

        return commandBuffer;
    }

    VkFence createComputeFence(VkDevice logicalDevice) {
        VkFence fence;
        VkFenceCreateInfo fenceCreateInfo = {};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        if (vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &fence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute fence");
        }

        return fence;
    }

    void extractComputeResult(VkDevice logicalDevice, VkDeviceMemory memory) {
        void* mappedMemory = NULL;

        if (vkMapMemory(logicalDevice, memory, 0, NUM_ELEMENTS * sizeof(glm::vec3), 0, &mappedMemory) != VK_SUCCESS) {
            throw std::runtime_error("Error mapping memory");
        }

        //float* floatMappedMemory = (float*) mappedMemory;
        //std::vector<float> nums(NUM_ELEMENTS);

        //glm::vec4* floatMappedMemory = (glm::vec4*) mappedMemory;
        //std::vector<glm::vec4> nums(NUM_ELEMENTS);

        glm::vec3* floatMappedMemory = (glm::vec3*) mappedMemory;
        std::vector<glm::vec3> nums(NUM_ELEMENTS);

        for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
            nums[i] = floatMappedMemory[i];
        }

        vkUnmapMemory(logicalDevice, memory);

        for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
            //std::cout << "i " << i << " n " << nums[i] << "\n";

            //glm::vec4 v = nums[i];
            //std::cout << "i " << i << " " << v.x << " " << v.y << " " << v.z << " " << v.w << "\n";

            glm::vec3 v = nums[i];
            std::cout << "i " << i << " " << v.x << " " << v.y << " " << v.z << "\n";
        }
    }

} // namespace anonymous

Simulator::Simulator(VkPhysicalDevice physicalDevice, VkDevice logicalDevice) {

    size_t computeQueueIndex = PhysicalDevice::findComputeQueueIndex(physicalDevice);
    vkGetDeviceQueue(logicalDevice, computeQueueIndex, 0, &m_computeQueue);

    m_computeDescriptorSetLayout = createComputeDescriptorSetLayout(logicalDevice);
    m_computeDescriptorPool = createComputeDescriptorPool(logicalDevice);

    Buffer::createBuffer(
        physicalDevice,
        logicalDevice,
        NUM_ELEMENTS * sizeof(Agent),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        m_computeBufferA,
        m_computeBufferMemoryA);

    Buffer::createBuffer(
        physicalDevice,
        logicalDevice,
        BUFFER_SIZE,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        m_computeBufferB,
        m_computeBufferMemoryB);

    Buffer::createBuffer(
        physicalDevice,
        logicalDevice,
        NUM_ELEMENTS * sizeof(glm::vec3),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        m_computeBufferC,
        m_computeBufferMemoryC);

    m_computeDescriptorSet = createComputeDescriptorSet(
        logicalDevice,
        m_computeDescriptorSetLayout,
        m_computeDescriptorPool,
        m_computeBufferA,
        m_computeBufferB,
        m_computeBufferC,
        BUFFER_SIZE);

    initializeComputeBuffers(logicalDevice, m_computeBufferMemoryA, m_computeBufferMemoryB);

    createComputePipeline(logicalDevice, m_computeDescriptorSetLayout, m_computePipelineLayout, m_computePipeline);

    m_computeCommandPool = createComputeCommandPool(physicalDevice, logicalDevice, computeQueueIndex);

    m_computeCommandBuffer = createComputeCommandBuffer(
        logicalDevice,
        m_computeCommandPool,
        m_computePipeline,
        m_computePipelineLayout,
        m_computeDescriptorSet);

    m_computeFence = createComputeFence(logicalDevice);
}

void Simulator::compute(VkDevice logicalDevice) {
    vkResetFences(logicalDevice, 1, &m_computeFence);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_computeCommandBuffer;

    if (vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_computeFence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit compute command buffer");
    }

    vkWaitForFences(logicalDevice, 1, &m_computeFence, VK_TRUE, UINT64_MAX);

    extractComputeResult(logicalDevice, m_computeBufferMemoryC);
}

void Simulator::cleanUp(VkDevice logicalDevice) {
    vkFreeMemory(logicalDevice, m_computeBufferMemoryA, nullptr);
    vkDestroyBuffer(logicalDevice, m_computeBufferA, nullptr);

    vkFreeMemory(logicalDevice, m_computeBufferMemoryB, nullptr);
    vkDestroyBuffer(logicalDevice, m_computeBufferB, nullptr);

    vkFreeMemory(logicalDevice, m_computeBufferMemoryC, nullptr);
    vkDestroyBuffer(logicalDevice, m_computeBufferC, nullptr);

    vkDestroyFence(logicalDevice, m_computeFence, nullptr);
    vkFreeCommandBuffers(logicalDevice, m_computeCommandPool, 1, &m_computeCommandBuffer);
    vkDestroyCommandPool(logicalDevice, m_computeCommandPool, nullptr);

    vkDestroyDescriptorPool(logicalDevice, m_computeDescriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(logicalDevice, m_computeDescriptorSetLayout, nullptr);
    vkDestroyPipelineLayout(logicalDevice, m_computePipelineLayout, nullptr);
    vkDestroyPipeline(logicalDevice, m_computePipeline, nullptr);
}
