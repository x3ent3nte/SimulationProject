#include <Simulator/InsertionSorter.h>

#include <Utils/Buffer.h>
#include <Utils/Utils.h>
#include <Utils/MyMath.h>
#include <Utils/Compute.h>
#include <Utils/Timer.h>

#include <vector>
#include <stdexcept>
#include <iostream>

InsertionSorter::InsertionSorter(
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
        numberOfElements * sizeof(InsertionSorterUtil::ValueAndIndex),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_valueAndIndexBuffer,
        m_valueAndIndexBufferMemory);

    uint32_t zero = 0;
    Buffer::createBufferWithData(
        &zero,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        logicalDevice,
        commandPool,
        queue,
        m_wasSwappedBuffer,
        m_wasSwappedBufferMemory);

    Buffer::createBuffer(
        physicalDevice,
        logicalDevice,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_wasSwappedBufferHostVisible,
        m_wasSwappedBufferMemoryHostVisible);

    m_currentNumberOfElements = numberOfElements;
    Buffer::createBufferWithData(
        &m_currentNumberOfElements,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        logicalDevice,
        commandPool,
        queue,
        m_numberOfElementsBuffer,
        m_numberOfElementsBufferMemory);

    Buffer::createBuffer(
        physicalDevice,
        logicalDevice,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_numberOfElementsBufferHostVisible,
        m_numberOfElementsBufferMemoryHostVisible);

    Buffer::createBufferWithData(
        &zero,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        logicalDevice,
        commandPool,
        queue,
        m_offsetOneBuffer,
        m_offsetOneBufferMemory);

    uint32_t offset = InsertionSorterUtil::xDim;
    Buffer::createBufferWithData(
        &offset,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        logicalDevice,
        commandPool,
        queue,
        m_offsetTwoBuffer,
        m_offsetTwoBufferMemory);

    m_descriptorSetLayout = InsertionSorterUtil::createDescriptorSetLayout(logicalDevice);
    m_descriptorPool = InsertionSorterUtil::createDescriptorPool(logicalDevice, 2);
    m_pipelineLayout = Compute::createPipelineLayout(logicalDevice, m_descriptorSetLayout);

    m_pipeline = InsertionSorterUtil::createPipeline(
        logicalDevice,
        m_pipelineLayout);

    m_descriptorSetOne = InsertionSorterUtil::createDescriptorSet(
        logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        m_valueAndIndexBuffer,
        m_wasSwappedBuffer,
        m_numberOfElementsBuffer,
        m_offsetOneBuffer,
        numberOfElements);

    m_descriptorSetTwo = InsertionSorterUtil::createDescriptorSet(
        logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        m_valueAndIndexBuffer,
        m_wasSwappedBuffer,
        m_numberOfElementsBuffer,
        m_offsetTwoBuffer,
        numberOfElements);

    m_commandBuffer = VK_NULL_HANDLE;
    createCommandBuffer(numberOfElements);

    m_setNumberOfElementsCommandBuffer = Buffer::recordCopyCommand(
        logicalDevice,
        commandPool,
        m_numberOfElementsBufferHostVisible,
        m_numberOfElementsBuffer,
        sizeof(uint32_t));

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute fence");
    }
}

InsertionSorter::~InsertionSorter() {
    vkFreeMemory(m_logicalDevice, m_valueAndIndexBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_valueAndIndexBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_wasSwappedBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_wasSwappedBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_wasSwappedBufferMemoryHostVisible, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_wasSwappedBufferHostVisible, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfElementsBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfElementsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfElementsBufferMemoryHostVisible, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfElementsBufferHostVisible, nullptr);

    vkFreeMemory(m_logicalDevice, m_offsetOneBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_offsetOneBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_offsetTwoBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_offsetTwoBuffer, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_descriptorSetLayout, nullptr);

    vkDestroyDescriptorPool(m_logicalDevice, m_descriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_pipeline, nullptr);

    std::array<VkCommandBuffer, 2> commandBuffers = {m_commandBuffer, m_setNumberOfElementsCommandBuffer};
    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, commandBuffers.size(), commandBuffers.data());

    vkDestroyFence(m_logicalDevice, m_fence, nullptr);
}

void InsertionSorter::setNumberOfElements(uint32_t numberOfElements) {

    if (m_currentNumberOfElements == numberOfElements) {
        return;
    }

    m_currentNumberOfElements = numberOfElements;

    createCommandBuffer(numberOfElements);

    void* dataMap;
    vkMapMemory(m_logicalDevice, m_numberOfElementsBufferMemoryHostVisible, 0, sizeof(uint32_t), 0, &dataMap);
    uint32_t numberOfElementsCopy = numberOfElements;
    memcpy(dataMap, &numberOfElementsCopy, sizeof(uint32_t));
    vkUnmapMemory(m_logicalDevice, m_numberOfElementsBufferMemoryHostVisible);

    VkSubmitInfo submitInfoOne{};
    submitInfoOne.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfoOne.commandBufferCount = 1;
    submitInfoOne.pCommandBuffers = &m_setNumberOfElementsCommandBuffer;

    vkResetFences(m_logicalDevice, 1, &m_fence);

    if (vkQueueSubmit(m_queue, 1, &submitInfoOne, m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit insertion sort set data size command buffer");
    }
    vkWaitForFences(m_logicalDevice, 1, &m_fence, VK_TRUE, UINT64_MAX);
}

void InsertionSorter::createCommandBuffer(uint32_t numberOfElements) {

    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &m_commandBuffer);
    m_commandBuffer = InsertionSorterUtil::createCommandBuffer(
        m_logicalDevice,
        m_commandPool,
        m_pipeline,
        m_pipelineLayout,
        m_descriptorSetOne,
        m_descriptorSetTwo,
        m_valueAndIndexBuffer,
        m_wasSwappedBuffer,
        m_wasSwappedBufferHostVisible,
        numberOfElements);
}

void InsertionSorter::setWasSwappedToZero() {
    void* dataMap;
    vkMapMemory(m_logicalDevice, m_wasSwappedBufferMemoryHostVisible, 0, sizeof(uint32_t), 0, &dataMap);
    uint32_t zero = 0;
    memcpy(dataMap, &zero, sizeof(uint32_t));
    vkUnmapMemory(m_logicalDevice, m_wasSwappedBufferMemoryHostVisible);
}

void InsertionSorter::runSortCommands() {
    VkSubmitInfo submitInfoOne{};
    submitInfoOne.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfoOne.commandBufferCount = 1;
    submitInfoOne.pCommandBuffers = &m_commandBuffer;

    vkResetFences(m_logicalDevice, 1, &m_fence);

    if (vkQueueSubmit(m_queue, 1, &submitInfoOne, m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit insertion sort command buffer");
    }
    vkWaitForFences(m_logicalDevice, 1, &m_fence, VK_TRUE, UINT64_MAX);
}

uint32_t InsertionSorter::needsSorting() {
    void* dataMap;
    vkMapMemory(m_logicalDevice, m_wasSwappedBufferMemoryHostVisible, 0, sizeof(uint32_t), 0, &dataMap);
    uint32_t wasSwappedValue = 0;
    memcpy(&wasSwappedValue, dataMap, sizeof(uint32_t));
    vkUnmapMemory(m_logicalDevice, m_wasSwappedBufferMemoryHostVisible);

    return wasSwappedValue;
}

void InsertionSorter::run(uint32_t numberOfElements) {

    int numIterations = 0;

    {
        //Timer timer("Insertion Sort Vulkan");

        setNumberOfElements(numberOfElements);

        do {
            setWasSwappedToZero();

            runSortCommands();

            numIterations += 1;
        } while (needsSorting());
    }

    //std::cout << "Insertion Sort Vulkan total number of iterations = " << numIterations << "\n";
}
