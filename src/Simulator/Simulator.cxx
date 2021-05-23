#include <Simulator/Simulator.h>

#include <Simulator/Agent.h>
#include <Simulator/Seeder.h>
#include <Utils/Buffer.h>
#include <Utils/Compute.h>
#include <Utils/Utils.h>
#include <Renderer/PhysicalDevice.h>
#include <Utils/MyGLM.h>
#include <Utils/Command.h>
#include <Utils/MyMath.h>

#include <Utils/Timer.h>

#include <array>
#include <stdexcept>
#include <iostream>

namespace {

    constexpr size_t simulatorXDim = 512;
    constexpr size_t kNumberOfBindings = 4;

    VkDescriptorSetLayout createComputeDescriptorSetLayout(VkDevice logicalDevice) {
        return Compute::createDescriptorSetLayout(logicalDevice, kNumberOfBindings);
    }

    VkDescriptorPool createComputeDescriptorPool(VkDevice logicalDevice) {
        return Compute::createDescriptorPool(logicalDevice, kNumberOfBindings, 1);
    }

    VkDescriptorSet createComputeDescriptorSet(
        VkDevice logicalDevice,
        VkDescriptorSetLayout& descriptorSetLayout,
        VkDescriptorPool& descriptorPool,
        VkBuffer agentsBuffer,
        VkBuffer positionsBuffer,
        VkBuffer timeDeltaBuffer,
        VkBuffer numberOfElementsBuffer,
        uint32_t maxNumberOfAgents) {

        std::vector<Compute::BufferAndSize> bufferAndSizes = {
            {agentsBuffer, maxNumberOfAgents * sizeof(Agent)},
            {positionsBuffer, maxNumberOfAgents * sizeof(AgentRenderInfo)},
            {timeDeltaBuffer, sizeof(float)},
            {numberOfElementsBuffer, sizeof(uint32_t)}
        };

        return Compute::createDescriptorSet(
            logicalDevice,
            descriptorSetLayout,
            descriptorPool,
            bufferAndSizes);
    }

    VkPipelineLayout createComputePipelineLayout(VkDevice logicalDevice, VkDescriptorSetLayout descriptorSetLayout) {
        return Compute::createPipelineLayout(logicalDevice, descriptorSetLayout);
    }

    VkPipeline createComputePipeline(
        VkDevice logicalDevice,
        VkDescriptorSetLayout descriptorSetLayout,
        VkPipelineLayout pipelineLayout) {

        return Compute::createPipeline("src/GLSL/spv/Simulation.spv", logicalDevice, pipelineLayout);
    }

    VkCommandBuffer createComputeCommandBuffer(
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkPipeline pipeline,
        VkPipelineLayout pipelineLayout,
        VkDescriptorSet descriptorSet,
        VkBuffer timeDeltaHostVisibleBuffer,
        VkBuffer timeDeltaBuffer,
        uint32_t numberOfElements) {

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

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = sizeof(float);
        vkCmdCopyBuffer(commandBuffer, timeDeltaHostVisibleBuffer, timeDeltaBuffer, 1, &copyRegion);

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            0,
            nullptr);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

        uint32_t xGroups = ceil(((float) numberOfElements) / ((float) simulatorXDim));
        vkCmdDispatch(commandBuffer, xGroups, 1, 1);

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
    std::shared_ptr<Connector> connector,
    std::shared_ptr<InputTerminal> inputTerminal,
    std::shared_ptr<Mesh> mesh,
    uint32_t maxNumberOfAgents,
    uint32_t maxNumberOfPlayers) {

    std::cout << "Size of Agent = " << sizeof(Agent) << "\n";

    m_mesh = mesh;

    m_logicalDevice = logicalDevice;
    m_computeQueue = computeQueue;
    m_computeCommandPool = computeCommandPool;

    m_currentNumberOfElements = maxNumberOfAgents / 2;

    m_isActive = false;
    m_connector = connector;
    m_inputTerminal = inputTerminal;

    const size_t numBuffers = m_connector->m_connections.size();

    m_computeDescriptorPools.resize(numBuffers);
    m_computeDescriptorSets.resize(numBuffers);

    m_computePipelines.resize(numBuffers);
    m_computePipelineLayouts.resize(numBuffers);
    m_computeCommandBuffers.resize(numBuffers);

    m_computeFence = createComputeFence(m_logicalDevice);

    std::vector<Agent> agents = Seeder::seed(m_currentNumberOfElements, maxNumberOfPlayers, m_mesh);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        maxNumberOfAgents * sizeof(Agent),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_agentsBuffer,
        m_agentsBufferMemory);

    Buffer::copyHostToDeviceBuffer(
        agents.data(),
        agents.size() * sizeof(Agent),
        m_agentsBuffer,
        physicalDevice,
        m_logicalDevice,
        m_computeCommandPool,
        m_computeQueue);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_timeDeltaBuffer,
        m_timeDeltaDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_timeDeltaBufferHostVisible,
        m_timeDeltaDeviceMemoryHostVisible);

    Buffer::createBufferWithData(
        &maxNumberOfAgents,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        m_logicalDevice,
        m_computeCommandPool,
        m_computeQueue,
        m_numberOfElementsBuffer,
        m_numberOfElementsDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_numberOfElementsBufferHostVisible,
        m_numberOfElementsDeviceMemoryHostVisible);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        maxNumberOfPlayers * sizeof(AgentRenderInfo),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_playerRenderInfosBuffer,
        m_playerRenderInfosDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        maxNumberOfPlayers * sizeof(AgentRenderInfo),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_playerRenderInfosHostVisibleBuffer,
        m_playerRenderInfosHostVisibleDeviceMemory);

    m_computeDescriptorSetLayout = createComputeDescriptorSetLayout(m_logicalDevice);

    for (size_t i = 0; i < numBuffers; ++i) {

        m_computeDescriptorPools[i] = createComputeDescriptorPool(m_logicalDevice);

        m_computeDescriptorSets[i] = createComputeDescriptorSet(
            m_logicalDevice,
            m_computeDescriptorSetLayout,
            m_computeDescriptorPools[i],
            m_agentsBuffer,
            m_connector->m_connections[i]->m_buffer,
            m_timeDeltaBuffer,
            m_numberOfElementsBuffer,
            maxNumberOfAgents);

        m_computePipelineLayouts[i] = createComputePipelineLayout(m_logicalDevice, m_computeDescriptorSetLayout);

        m_computePipelines[i] = createComputePipeline(m_logicalDevice, m_computeDescriptorSetLayout, m_computePipelineLayouts[i]);

        m_computeCommandBuffers[i] = createComputeCommandBuffer(
            m_logicalDevice,
            m_computeCommandPool,
            m_computePipelines[i],
            m_computePipelineLayouts[i],
            m_computeDescriptorSets[i],
            m_timeDeltaBufferHostVisible,
            m_timeDeltaBuffer,
            maxNumberOfAgents);
    }

    m_collider = std::make_shared<Collider>(
        physicalDevice,
        m_logicalDevice,
        m_computeQueue,
        m_computeCommandPool,
        m_agentsBuffer,
        maxNumberOfAgents);

    m_agentSorter = std::make_shared<AgentSorter>(
        physicalDevice,
        m_logicalDevice,
        m_computeQueue,
        m_computeCommandPool,
        m_agentsBuffer,
        maxNumberOfAgents,
        false);

    m_boids = std::make_shared<Boids>(
        physicalDevice,
        m_logicalDevice,
        m_computeQueue,
        m_computeCommandPool,
        m_agentsBuffer,
        maxNumberOfAgents,
        maxNumberOfPlayers);

    auto simulationStateWriter = std::make_shared<SimulationStateWriter>(m_logicalDevice, m_connector->m_connections.size());
    for (size_t i = 0; i < numBuffers; ++i) {
        m_simulationStateWriterFunctions.push_back(
            std::make_shared<SimulationStateWriterFunction>(
                simulationStateWriter,
                m_agentsBuffer,
                m_connector->m_connections[i]->m_buffer,
                m_playerRenderInfosBuffer,
                maxNumberOfAgents,
                maxNumberOfPlayers));
    }
}

Simulator::~Simulator() {
    vkFreeMemory(m_logicalDevice, m_agentsBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_agentsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_timeDeltaDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_timeDeltaBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_timeDeltaDeviceMemoryHostVisible, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_timeDeltaBufferHostVisible, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfElementsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfElementsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfElementsDeviceMemoryHostVisible, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfElementsBufferHostVisible, nullptr);

    vkFreeMemory(m_logicalDevice, m_playerRenderInfosDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_playerRenderInfosBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_playerRenderInfosHostVisibleDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_playerRenderInfosHostVisibleBuffer, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_computeDescriptorSetLayout, nullptr);

    for (size_t i = 0; i < m_computePipelines.size(); ++i) {
        vkFreeCommandBuffers(m_logicalDevice, m_computeCommandPool, 1, &m_computeCommandBuffers[i]);

        vkDestroyDescriptorPool(m_logicalDevice, m_computeDescriptorPools[i], nullptr);
        vkDestroyPipelineLayout(m_logicalDevice, m_computePipelineLayouts[i], nullptr);
        vkDestroyPipeline(m_logicalDevice, m_computePipelines[i], nullptr);
    }

    vkDestroyFence(m_logicalDevice, m_computeFence, nullptr);
}

void Simulator::simulateNextStep(VkCommandBuffer commandBuffer, float timeDelta) {
    //Timer timer("Simulator::simulateNextStep");

    Buffer::writeHostVisible(&timeDelta, m_timeDeltaDeviceMemoryHostVisible, 0, sizeof(float), m_logicalDevice);

    Command::runAndWait(commandBuffer, m_computeFence, m_computeQueue, m_logicalDevice);
}

void Simulator::runSimulatorStateWriterFunction(uint32_t numberOfPlayers) {
    //Timer timer("Simulator::runSimulatorStateWriterFunction");

    auto connection = m_connector->takeOldConnection();
    auto sswf = m_simulationStateWriterFunctions[connection->m_id];

    VkCommandBuffer commandBuffer;

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = m_computeCommandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(m_logicalDevice, &commandBufferAllocateInfo, &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute command buffer");
    }

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin compute command buffer");
    }

    sswf->recordCommand(commandBuffer, m_currentNumberOfElements);

    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.pNext = nullptr;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &memoryBarrier,
        0,
        nullptr,
        0,
        nullptr);

    size_t playerMemorySize = numberOfPlayers * sizeof(AgentRenderInfo);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = playerMemorySize;
    vkCmdCopyBuffer(commandBuffer, m_playerRenderInfosBuffer, m_playerRenderInfosHostVisibleBuffer, 1, &copyRegion);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end compute command buffer");
    }

    Command::runAndWait(commandBuffer, m_computeFence, m_computeQueue, m_logicalDevice);

    vkFreeCommandBuffers(m_logicalDevice, m_computeCommandPool, 1, &commandBuffer);

    connection->m_players.resize(numberOfPlayers);
    Buffer::readHostVisible(m_playerRenderInfosHostVisibleDeviceMemory, connection->m_players.data(), 0, playerMemorySize, m_logicalDevice);

    connection->m_numberOfElements = m_currentNumberOfElements;
    m_connector->restoreNewestConnection(connection);
}

void Simulator::runSimulatorTask() {
    Timer timer("Vulkan Simulator");
    uint64_t numFrames = 0;

    auto prevTime = std::chrono::high_resolution_clock::now();

    while (m_isActive) {

        auto currentTime = std::chrono::high_resolution_clock::now();
        float timeDelta = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - prevTime).count();
        timeDelta = fmin(timeDelta, 0.01);
        //std::cout << "Time Delta= " << timeDelta << "\n";
        //Timer timer("Frame " + std::to_string(numFrames));

        m_collider->run(timeDelta, m_currentNumberOfElements);
        m_agentSorter->run(timeDelta, m_currentNumberOfElements);

        const std::vector<InputState> inputStates = m_inputTerminal->readInputStates();
        std::vector<uint32_t> inputStatesInt(inputStates.size());
        for (int i = 0; i < inputStates.size(); ++i) {
            inputStatesInt[i] = inputStates[i].m_state;
        }

        m_currentNumberOfElements = m_boids->run(timeDelta, m_currentNumberOfElements, inputStatesInt);
        std::cout << "New number of elements = " << m_currentNumberOfElements << "\n";

        //updateConnector(timeDelta);
        runSimulatorStateWriterFunction(inputStatesInt.size());

        numFrames++;
        prevTime = currentTime;
    }
    std::cout << "Number of frames simulated = " << numFrames << "\n";
}

void Simulator::simulate() {
    m_isActive = true;
    m_simulateTask = std::thread(&Simulator::runSimulatorTask, this);
}

void Simulator::stopSimulation(VkPhysicalDevice physicalDevice) {
    m_isActive = false;
    m_simulateTask.join();

    std::vector<Agent> agents(m_currentNumberOfElements);

    Buffer::copyDeviceBufferToHost(
        agents.data(),
        m_currentNumberOfElements * sizeof(Agent),
        m_agentsBuffer,
        physicalDevice,
        m_logicalDevice,
        m_computeCommandPool,
        m_computeQueue);

    for (size_t i = 0; i < m_currentNumberOfElements; ++i) {
        //glm::vec3 position = agents[i].position;
        //std::cout << "i " << i << " " << position.x << " " << position.y << " " << position.z << "\n";
        //glm::vec3 acceleration = agents[i].acceleration;
        //std::cout << "Acceleration Mag: " << glm::length(acceleration) << "\n";
    }
}
