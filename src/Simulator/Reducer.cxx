#include <Simulator/Reducer.h>

#include <Simulator/Collision.h>
#include <Simulator/ReducerUtil.h>
#include <Utils/Buffer.h>
#include <Utils/Compute.h>
#include <Renderer/Command.h>

#include <array>
#include <stdexcept>

#include <math.h>

Reducer::Reducer(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t numberOfElements) {

    m_physicalDevice = physicalDevice;
    m_logicalDevice = logicalDevice;
    m_queue = queue;
    m_commandPool = commandPool;

    Buffer::createBuffer(
        physicalDevice,
        logicalDevice,
        numberOfElements * sizeof(Collision),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_oneBuffer,
        m_oneBufferMemory);

    Buffer::createBuffer(
        physicalDevice,
        logicalDevice,
        numberOfElements * sizeof(Collision),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_twoBuffer,
        m_twoBufferMemory);

    Buffer::createBufferWithData(
        &numberOfElements,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        logicalDevice,
        commandPool,
        queue,
        m_numberOfElementsBuffer,
        m_numberOfElementsBufferMemory);

    Buffer::createBuffer(
        m_physicalDevice,
        m_logicalDevice,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_numberOfElementsBufferHostVisible,
        m_numberOfElementsBufferMemoryHostVisible);

    m_descriptorSetLayout = ReducerUtil::createDescriptorSetLayout(logicalDevice);
    m_descriptorPool = ReducerUtil::createDescriptorPool(logicalDevice, 2);
    m_pipelineLayout = Compute::createPipelineLayout(logicalDevice, m_descriptorSetLayout);

    m_pipeline = ReducerUtil::createPipeline(
        logicalDevice,
        m_pipelineLayout);

    m_oneToTwo = ReducerUtil::createDescriptorSet(
        m_logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        m_oneBuffer,
        m_twoBuffer,
        m_numberOfElementsBuffer,
        numberOfElements);
    m_twoToOne = ReducerUtil::createDescriptorSet(
        m_logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        m_twoBuffer,
        m_oneBuffer,
        m_numberOfElementsBuffer,
        numberOfElements);

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute fence");
    }
}

Reducer::~Reducer() {
    vkFreeMemory(m_logicalDevice, m_oneBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_oneBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_twoBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_twoBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfElementsBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfElementsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfElementsBufferMemoryHostVisible, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfElementsBufferHostVisible, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_descriptorSetLayout, nullptr);

    vkDestroyDescriptorPool(m_logicalDevice, m_descriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_pipeline, nullptr);

    vkDestroyFence(m_logicalDevice, m_fence, nullptr);
}

void Reducer::runReduceCommand(uint32_t numberOfElements, VkDescriptorSet descriptorSet) {

    Buffer::writeHostVisible(&numberOfElements, m_numberOfElementsBufferMemoryHostVisible, 0, sizeof(uint32_t), m_logicalDevice);

    VkCommandBuffer commandBuffer = ReducerUtil::createCommandBuffer(
        m_logicalDevice,
        m_commandPool,
        m_pipeline,
        m_pipelineLayout,
        descriptorSet,
        m_numberOfElementsBuffer,
        m_numberOfElementsBufferHostVisible,
        numberOfElements);

    Command::runAndWait(commandBuffer, m_fence, m_queue, m_logicalDevice);

    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &commandBuffer);
}

VkBuffer Reducer::run(uint32_t numberOfElements) {

    VkBuffer currentInput = m_twoBuffer;
    VkBuffer currentOutput = m_oneBuffer;

    VkDescriptorSet currentDescriptorSet = m_oneToTwo;
    VkDescriptorSet otherDescriptorSet = m_twoToOne;

    {
        while (numberOfElements > 1) {

            runReduceCommand(numberOfElements, currentDescriptorSet);

            numberOfElements = ceil(float(numberOfElements) / float(ReducerUtil::xDim * 2));

            VkBuffer tempBuffer = currentInput;
            currentInput = currentOutput;
            currentOutput = tempBuffer;

            VkDescriptorSet tempDescriptorSet = currentDescriptorSet;
            currentDescriptorSet = otherDescriptorSet;
            otherDescriptorSet = tempDescriptorSet;
        }
    }

    return currentOutput;
}
