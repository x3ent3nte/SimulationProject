#include <Simulator/Simulator.h>

#include <Simulator/Agent.h>
#include <Utils/Buffer.h>
#include <Utils/Utils.h>
#include <Renderer/PhysicalDevice.h>
#include <Renderer/MyGLM.h>
#include <Utils/MyMath.h>
#include <Renderer/Constants.h>

#include <Utils/Timer.h>

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

        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[0].descriptorCount = 1;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = 1;

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
        descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.maxSets = 1;
        descriptorPoolCreateInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());;
        descriptorPoolCreateInfo.pPoolSizes = poolSizes.data();

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

Simulator::Simulator(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue computeQueue,
    VkCommandPool computeCommandPool,
    std::shared_ptr<Connector> connector) {

    m_logicalDevice = logicalDevice;
    m_computeQueue = computeQueue;
    m_computeCommandPool = computeCommandPool;

    m_isActive = false;
    m_connector = connector;

    const size_t numBuffers = m_connector->m_buffers.size();

    m_computeDescriptorPools.resize(numBuffers);
    m_computeDescriptorSets.resize(numBuffers);

    m_computePipelines.resize(numBuffers);
    m_computePipelineLayouts.resize(numBuffers);
    m_computeCommandBuffers.resize(numBuffers);

    m_computeFence = createComputeFence(m_logicalDevice);

    std::vector<Agent> agents(Constants::kNumberOfAgents);
    for (size_t i = 0; i < Constants::kNumberOfAgents; ++i) {
        glm::vec3 position = MyMath::randomVec3InSphere(512.0f);
        glm::vec3 velocity = glm::vec3{0.0f, 0.0f, 0.0f};
        glm::vec3 acceleration = glm::vec3(0.0f, 0.0f, 0.0f);
        glm::vec3 target = MyMath::randomVec3InSphere(256.f) + position;
        glm::vec4 rotation = MyMath::createQuaternionFromAxisAndTheta(glm::vec3(0.0f), 0.0f);
        agents[i] = Agent{position, velocity, acceleration, target, rotation};
    }

    Buffer::createBufferWithData(
        agents.data(),
        Constants::kNumberOfAgents * sizeof(Agent),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        m_logicalDevice,
        m_computeCommandPool,
        m_computeQueue,
        m_agentsBuffer,
        m_agentsBufferMemory);

    m_computeDescriptorSetLayout = createComputeDescriptorSetLayout(m_logicalDevice);

    auto shaderCode = Utils::readFile("src/GLSL/Simulation.spv");
    VkShaderModule shaderModule = Utils::createShaderModule(m_logicalDevice, shaderCode);

    for (size_t i = 0; i < numBuffers; ++i) {

        m_computeDescriptorPools[i] = createComputeDescriptorPool(m_logicalDevice);

        m_computeDescriptorSets[i] = createComputeDescriptorSet(
            m_logicalDevice,
            m_computeDescriptorSetLayout,
            m_computeDescriptorPools[i],
            m_agentsBuffer,
            m_connector->m_buffers[i]);

        m_computePipelineLayouts[i] = createComputePipelineLayout(m_logicalDevice, m_computeDescriptorSetLayout);

        m_computePipelines[i] = createComputePipeline(m_logicalDevice, shaderModule, m_computeDescriptorSetLayout, m_computePipelineLayouts[i]);

        m_computeCommandBuffers[i] = createComputeCommandBuffer(
            m_logicalDevice,
            m_computeCommandPool,
            m_computePipelines[i],
            m_computePipelineLayouts[i],
            m_computeDescriptorSets[i]);
    }

    vkDestroyShaderModule(m_logicalDevice, shaderModule, nullptr);

    m_agentSorter = std::make_shared<AgentSorter>(
        physicalDevice,
        m_logicalDevice,
        m_computeQueue,
        m_computeCommandPool,
        m_agentsBuffer,
        Constants::kNumberOfAgents);
}

Simulator::~Simulator() {
    vkFreeMemory(m_logicalDevice, m_agentsBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_agentsBuffer, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_computeDescriptorSetLayout, nullptr);

    for (size_t i = 0; i < m_computePipelines.size(); ++i) {
        vkFreeCommandBuffers(m_logicalDevice, m_computeCommandPool, 1, &m_computeCommandBuffers[i]);

        vkDestroyDescriptorPool(m_logicalDevice, m_computeDescriptorPools[i], nullptr);
        vkDestroyPipelineLayout(m_logicalDevice, m_computePipelineLayouts[i], nullptr);
        vkDestroyPipeline(m_logicalDevice, m_computePipelines[i], nullptr);
    }

    vkDestroyFence(m_logicalDevice, m_computeFence, nullptr);
}

void Simulator::simulateNextStep(VkCommandBuffer commandBuffer) {
    vkResetFences(m_logicalDevice, 1, &m_computeFence);


    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    if (vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_computeFence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit compute command buffer");
    }

    vkWaitForFences(m_logicalDevice, 1, &m_computeFence, VK_TRUE, UINT64_MAX);
}

void Simulator::runSimulatorTask() {
    Timer timer("Vulkan Simulator");
    uint64_t numFrames = 0;

    while (m_isActive) {
        //Timer timer("Frame " + std::to_string(numFrames));
        size_t bufferIndex = m_connector->takeOldBufferIndex();
        m_agentSorter->run(0.09, Constants::kNumberOfAgents);
        simulateNextStep(m_computeCommandBuffers[bufferIndex]);
        m_connector->updateBufferIndex(bufferIndex);

        numFrames++;
    }
    std::cout << "Number of frames simulated: " << numFrames << "\n";
}

void Simulator::simulate() {
    m_isActive = true;
    m_simulateTask = std::thread(&Simulator::runSimulatorTask, this);
}

void Simulator::stopSimulation(VkPhysicalDevice physicalDevice) {
    m_isActive = false;
    m_simulateTask.join();

    std::vector<Agent> agents(Constants::kNumberOfAgents);

    Buffer::copyDeviceBufferToHost(
        agents.data(),
        Constants::kNumberOfAgents * sizeof(Agent),
        m_agentsBuffer,
        physicalDevice,
        m_logicalDevice,
        m_computeCommandPool,
        m_computeQueue);

    for (size_t i = 0; i < Constants::kNumberOfAgents; ++i) {
        //glm::vec3 position = agents[i].position;
        //std::cout << "i " << i << " " << position.x << " " << position.y << " " << position.z << "\n";
        glm::vec3 acceleration = agents[i].acceleration;
        //std::cout << "Acceleration Mag: " << glm::length(acceleration) << "\n";
    }
}
