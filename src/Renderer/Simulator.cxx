#include <Renderer/Simulator.h>

#include <Renderer/Buffer.h>
#include <Renderer/Utils.h>
#include <Renderer/PhysicalDevice.h>
#include <Renderer/MyGLM.h>
#include <Renderer/MyMath.h>

#include <Timer.h>

#include <array>
#include <stdexcept>
#include <iostream>

#define X_DIM 512
#define NUM_ELEMENTS 32 * X_DIM

namespace {

    VkDescriptorSetLayout createComputeDescriptorSetLayout(VkDevice logicalDevice) {
        VkDescriptorSetLayout descriptorSetLayout;

        VkDescriptorSetLayoutBinding agentDescriptor = {};
        agentDescriptor.binding = 0;
        agentDescriptor.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        agentDescriptor.descriptorCount = 1;
        agentDescriptor.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding positionDescriptor = {};
        positionDescriptor.binding = 1;
        positionDescriptor.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        positionDescriptor.descriptorCount = 1;
        positionDescriptor.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        std::array<VkDescriptorSetLayoutBinding, 2> descriptorSetLayoutBindings = {agentDescriptor, positionDescriptor};

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.bindingCount = 2;
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
        descriptorPoolSize.descriptorCount = 2;

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
        glm::vec3 position;
        glm::vec3 target;
    };

    void initializeComputeBuffers(VkDevice logicalDevice, VkDeviceMemory memoryA) {
        void* mappedMemoryA = NULL;
        vkMapMemory(logicalDevice, memoryA, 0, NUM_ELEMENTS * sizeof(Agent), 0, & mappedMemoryA);
        Agent* floatMappedMemoryA = (Agent*) mappedMemoryA;
        for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
            glm::vec3 position = MyMath::randomVec3InSphere(4096.0f);
            glm::vec3 target = MyMath::randomVec3InSphere(2048.f) + position;
            //glm::vec3 position = glm::vec3(i);
            //glm::vec3 target = glm::vec3(i + 1);
            floatMappedMemoryA[i] = {position, target};
        }
        vkUnmapMemory(logicalDevice, memoryA);
    }

    VkDescriptorSet createComputeDescriptorSet(
        VkDevice logicalDevice,
        VkDescriptorSetLayout& descriptorSetLayout ,
        VkDescriptorPool& descriptorPool,
        VkBuffer agentsBuffer,
        VkBuffer positionsBuffer) {

        VkDescriptorSet descriptorSet;

        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
        descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocateInfo.descriptorPool = descriptorPool;
        descriptorSetAllocateInfo.descriptorSetCount = 1;
        descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

        if (vkAllocateDescriptorSets(logicalDevice, &descriptorSetAllocateInfo, &descriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute descriptor sets");
        }

        VkDescriptorBufferInfo agentsBufferDescriptor = {};
        agentsBufferDescriptor.buffer = agentsBuffer;
        agentsBufferDescriptor.offset = 0;
        agentsBufferDescriptor.range = NUM_ELEMENTS * sizeof(Agent);

        VkWriteDescriptorSet agentsWriteDescriptorSet = {};
        agentsWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        agentsWriteDescriptorSet.dstSet = descriptorSet;
        agentsWriteDescriptorSet.dstBinding = 0;
        agentsWriteDescriptorSet.descriptorCount = 1;
        agentsWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        agentsWriteDescriptorSet.pBufferInfo = &agentsBufferDescriptor;

        VkDescriptorBufferInfo positionsBufferDescriptor = {};
        positionsBufferDescriptor.buffer = positionsBuffer;
        positionsBufferDescriptor.offset = 0;
        positionsBufferDescriptor.range = NUM_ELEMENTS * sizeof(glm::vec3);

        VkWriteDescriptorSet positionsWriteDescriptorSet = {};
        positionsWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        positionsWriteDescriptorSet.dstSet = descriptorSet;
        positionsWriteDescriptorSet.dstBinding = 1;
        positionsWriteDescriptorSet.descriptorCount = 1;
        positionsWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        positionsWriteDescriptorSet.pBufferInfo = &positionsBufferDescriptor;

        std::array<VkWriteDescriptorSet, 2> writeDescriptorSets = {agentsWriteDescriptorSet, positionsWriteDescriptorSet};

        vkUpdateDescriptorSets(logicalDevice, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, nullptr);

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
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

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
} // namespace anonymous

Simulator::Simulator(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, std::shared_ptr<Connector> connector) {

    m_isActive = false;
    m_connector = connector;

    //size_t computeQueueIndex = PhysicalDevice::findComputeQueueIndex(physicalDevice);
    size_t computeQueueIndex = 2;
    vkGetDeviceQueue(logicalDevice, computeQueueIndex, 0, &m_computeQueue);

    m_computeCommandPool = createComputeCommandPool(physicalDevice, logicalDevice, computeQueueIndex);

    m_computeDescriptorSetLayout = createComputeDescriptorSetLayout(logicalDevice);
    m_computeDescriptorPool = createComputeDescriptorPool(logicalDevice);

    std::vector<Agent> agents(NUM_ELEMENTS);
    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        glm::vec3 position = MyMath::randomVec3InSphere(4096.0f);
        glm::vec3 target = MyMath::randomVec3InSphere(2048.f) + position;
        agents[i] = Agent{position, target};
    }

    Buffer::createReadOnlyBuffer(
        agents.data(),
        NUM_ELEMENTS * sizeof(Agent),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        logicalDevice,
        m_computeCommandPool,
        m_computeQueue,
        m_agentsBuffer,
        m_agentsBufferMemory);

    std::vector<glm::vec3> positions(NUM_ELEMENTS);
    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        positions[i] = glm::vec3(0);
    }

    Buffer::createReadOnlyBuffer(
        positions.data(),
        NUM_ELEMENTS * sizeof(glm::vec3),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        logicalDevice,
        m_computeCommandPool,
        m_computeQueue,
        m_positionsBuffer,
        m_positionsBufferMemory);

    m_computeDescriptorSet = createComputeDescriptorSet(
        logicalDevice,
        m_computeDescriptorSetLayout,
        m_computeDescriptorPool,
        m_agentsBuffer,
        m_positionsBuffer);

    createComputePipeline(logicalDevice, m_computeDescriptorSetLayout, m_computePipelineLayout, m_computePipeline);

    m_computeCommandBuffer = createComputeCommandBuffer(
        logicalDevice,
        m_computeCommandPool,
        m_computePipeline,
        m_computePipelineLayout,
        m_computeDescriptorSet);

    m_computeFence = createComputeFence(logicalDevice);
}

void Simulator::simulateNextStep(VkDevice logicalDevice) {
    vkResetFences(logicalDevice, 1, &m_computeFence);

    size_t numCommands = 1;
    std::vector<VkSubmitInfo> submitInfos(numCommands);
    {
        for (size_t  j = 0; j < numCommands; ++j) {
            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &m_computeCommandBuffer;

            submitInfos[j] = submitInfo;
        }
    }

    if (vkQueueSubmit(m_computeQueue, submitInfos.size(), submitInfos.data(), m_computeFence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit compute command buffer");
    }

    vkWaitForFences(logicalDevice, 1, &m_computeFence, VK_TRUE, UINT64_MAX);
}

void Simulator::runSimulatorTask(VkDevice logicalDevice) {
    Timer time("Vulkan Simulator");
    uint64_t numFrames = 0;
    while (m_isActive) {
        simulateNextStep(logicalDevice);
        numFrames++;
    }
    std::cout << "Number of frames simulated: " << numFrames << "\n";
}

void Simulator::simulate(VkDevice logicalDevice) {
    m_isActive = true;
    m_simulateTask = std::thread(&Simulator::runSimulatorTask, this, logicalDevice);
}

void Simulator::stopSimulation(VkPhysicalDevice physicalDevice, VkDevice logicalDevice) {
    m_isActive = false;
    m_simulateTask.join();

    std::vector<glm::vec3> positions(NUM_ELEMENTS);

    Buffer::copyDeviceBufferToHost(
        positions.data(),
        NUM_ELEMENTS * sizeof(glm::vec3),
        m_positionsBuffer,
        physicalDevice,
        logicalDevice,
        m_computeCommandPool,
        m_computeQueue);

    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        glm::vec3 position = positions[i];
        //std::cout << "i " << i << " " << position.x << " " << position.y << " " << position.z << "\n";
    }
}

void Simulator::cleanUp(VkDevice logicalDevice) {

    vkFreeMemory(logicalDevice, m_agentsBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, m_agentsBuffer, nullptr);

    vkFreeMemory(logicalDevice, m_positionsBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, m_positionsBuffer, nullptr);

    vkDestroyFence(logicalDevice, m_computeFence, nullptr);
    vkFreeCommandBuffers(logicalDevice, m_computeCommandPool, 1, &m_computeCommandBuffer);
    vkDestroyCommandPool(logicalDevice, m_computeCommandPool, nullptr);

    vkDestroyDescriptorPool(logicalDevice, m_computeDescriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(logicalDevice, m_computeDescriptorSetLayout, nullptr);
    vkDestroyPipelineLayout(logicalDevice, m_computePipelineLayout, nullptr);
    vkDestroyPipeline(logicalDevice, m_computePipeline, nullptr);
}
