#include <Renderer/Simulator.h>

#include <Renderer/Agent.h>
#include <Renderer/Buffer.h>
#include <Renderer/Utils.h>
#include <Renderer/PhysicalDevice.h>
#include <Renderer/MyGLM.h>
#include <Renderer/MyMath.h>
#include <Renderer/Constants.h>

#include <Timer.h>

#include <array>
#include <stdexcept>
#include <iostream>

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

    VkDescriptorSet createComputeDescriptorSet(
        VkDevice logicalDevice,
        VkDescriptorSetLayout& descriptorSetLayout,
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
        agentsBufferDescriptor.range = Constants::kNumberOfAgents * sizeof(Agent);

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
        positionsBufferDescriptor.range = Constants::kNumberOfAgents * sizeof(AgentPositionAndRotation);

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

    VkPipelineLayout createComputePipelineLayout(VkDevice logicalDevice, VkDescriptorSetLayout descriptorSetLayout) {
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

    VkPipeline createComputePipeline(
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

        size_t xThreads = Constants::kNumberOfAgents / 512;
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

    const size_t numBuffers = m_connector->m_buffers.size();

    m_computeDescriptorPools.resize(numBuffers);
    m_computeDescriptorSets.resize(numBuffers);

    m_computePipelines.resize(numBuffers);
    m_computePipelineLayouts.resize(numBuffers);
    m_computeCommandBuffers.resize(numBuffers);

    //size_t computeQueueIndex = PhysicalDevice::findComputeQueueIndex(physicalDevice);
    size_t computeQueueIndex = 2;
    vkGetDeviceQueue(logicalDevice, computeQueueIndex, 0, &m_computeQueue);

    m_computeCommandPool = createComputeCommandPool(physicalDevice, logicalDevice, computeQueueIndex);

    m_computeFence = createComputeFence(logicalDevice);

    std::vector<Agent> agents(Constants::kNumberOfAgents);
    for (size_t i = 0; i < Constants::kNumberOfAgents; ++i) {
        glm::vec3 position = MyMath::randomVec3InSphere(512.0f);
        glm::vec3 velocity = glm::vec3{0.0f, 0.0f, 0.0f};
        glm::vec3 target = MyMath::randomVec3InSphere(256.f) + position;
        glm::vec4 rotation = MyMath::createQuaternionFromAxisAndTheta(glm::vec3(0.0f), 0.0f);
        agents[i] = Agent{position, velocity, target, rotation};
    }

    Buffer::createReadOnlyBuffer(
        agents.data(),
        Constants::kNumberOfAgents * sizeof(Agent),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        logicalDevice,
        m_computeCommandPool,
        m_computeQueue,
        m_agentsBuffer,
        m_agentsBufferMemory);

    m_computeDescriptorSetLayout = createComputeDescriptorSetLayout(logicalDevice);

    auto shaderCode = Utils::readFile("src/GLSL/kernel.spv");
    VkShaderModule shaderModule = Utils::createShaderModule(logicalDevice, shaderCode);

    for (size_t i = 0; i < numBuffers; ++i) {

        m_computeDescriptorPools[i] = createComputeDescriptorPool(logicalDevice);

        m_computeDescriptorSets[i] = createComputeDescriptorSet(
            logicalDevice,
            m_computeDescriptorSetLayout,
            m_computeDescriptorPools[i],
            m_agentsBuffer,
            m_connector->m_buffers[i]);

        m_computePipelineLayouts[i] = createComputePipelineLayout(logicalDevice, m_computeDescriptorSetLayout);

        m_computePipelines[i] = createComputePipeline(logicalDevice, shaderModule, m_computeDescriptorSetLayout, m_computePipelineLayouts[i]);

        m_computeCommandBuffers[i] = createComputeCommandBuffer(
            logicalDevice,
            m_computeCommandPool,
            m_computePipelines[i],
            m_computePipelineLayouts[i],
            m_computeDescriptorSets[i]);
    }

    vkDestroyShaderModule(logicalDevice, shaderModule, nullptr);
}

void Simulator::simulateNextStep(VkDevice logicalDevice, VkCommandBuffer commandBuffer) {
    vkResetFences(logicalDevice, 1, &m_computeFence);


    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    if (vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_computeFence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit compute command buffer");
    }

    vkWaitForFences(logicalDevice, 1, &m_computeFence, VK_TRUE, UINT64_MAX);
}

void Simulator::runSimulatorTask(VkDevice logicalDevice) {
    Timer time("Vulkan Simulator");
    uint64_t numFrames = 0;

    while (m_isActive) {
        size_t bufferIndex = m_connector->takeOldBufferIndex();
        //std::cout << "Updating buffer index " << bufferIndex << "\n";
        simulateNextStep(logicalDevice, m_computeCommandBuffers[bufferIndex]);
        m_connector->updateBufferIndex(bufferIndex);

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

    std::vector<AgentPositionAndRotation> positions(Constants::kNumberOfAgents);

    Buffer::copyDeviceBufferToHost(
        positions.data(),
        Constants::kNumberOfAgents * sizeof(AgentPositionAndRotation),
        m_connector->m_buffers[2],
        physicalDevice,
        logicalDevice,
        m_computeCommandPool,
        m_computeQueue);

    for (size_t i = 0; i < Constants::kNumberOfAgents; ++i) {
        glm::vec3 position = positions[i].position;
        //std::cout << "i " << i << " " << position.x << " " << position.y << " " << position.z << "\n";
    }
}

void Simulator::cleanUp(VkDevice logicalDevice) {

    vkFreeMemory(logicalDevice, m_agentsBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, m_agentsBuffer, nullptr);

    vkDestroyDescriptorSetLayout(logicalDevice, m_computeDescriptorSetLayout, nullptr);

    for (size_t i = 0; i < m_computePipelines.size(); ++i) {
        vkFreeCommandBuffers(logicalDevice, m_computeCommandPool, 1, &m_computeCommandBuffers[i]);

        vkDestroyDescriptorPool(logicalDevice, m_computeDescriptorPools[i], nullptr);
        vkDestroyPipelineLayout(logicalDevice, m_computePipelineLayouts[i], nullptr);
        vkDestroyPipeline(logicalDevice, m_computePipelines[i], nullptr);
    }

    vkDestroyCommandPool(logicalDevice, m_computeCommandPool, nullptr);
    vkDestroyFence(logicalDevice, m_computeFence, nullptr);
}
