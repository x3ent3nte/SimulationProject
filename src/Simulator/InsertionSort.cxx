#include <Simulator/InsertionSort.h>

#include <Utils/Buffer.h>
#include <Utils/Utils.h>
#include <Utils/MyMath.h>
#include <Utils/Compute.h>
#include <Utils/Timer.h>

#include <vector>
#include <stdexcept>
#include <iostream>

InsertionSort::InsertionSort(
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
        numberOfElements * sizeof(InsertionSortUtil::ValueAndIndex),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_valueAndIndexBuffer,
        m_valueAndIndexBufferMemory);

    uint32_t zero = 0;
    Buffer::createReadOnlyBuffer(
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

    m_currentDataSize = numberOfElements;
    Buffer::createReadOnlyBuffer(
        &m_currentDataSize,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        logicalDevice,
        commandPool,
        queue,
        m_dataSizeBuffer,
        m_dataSizeBufferMemory);

    Buffer::createBuffer(
        physicalDevice,
        logicalDevice,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_dataSizeBufferHostVisible,
        m_dataSizeBufferMemoryHostVisible);

    Buffer::createReadOnlyBuffer(
        &zero,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        logicalDevice,
        commandPool,
        queue,
        m_offsetOneBuffer,
        m_offsetOneBufferMemory);

    uint32_t offset = X_DIM;
    Buffer::createReadOnlyBuffer(
        &offset,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        logicalDevice,
        commandPool,
        queue,
        m_offsetTwoBuffer,
        m_offsetTwoBufferMemory);

    m_descriptorSetLayout = InsertionSortUtil::createDescriptorSetLayout(logicalDevice);
    m_descriptorPool = InsertionSortUtil::createDescriptorPool(logicalDevice, 2);
    m_pipelineLayout = Compute::createPipelineLayout(logicalDevice, m_descriptorSetLayout);

    m_pipeline = InsertionSortUtil::createPipeline(
        logicalDevice,
        m_pipelineLayout);

    m_descriptorSetOne = InsertionSortUtil::createDescriptorSet(
        logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        m_valueAndIndexBuffer,
        m_wasSwappedBuffer,
        m_dataSizeBuffer,
        m_offsetOneBuffer,
        numberOfElements);

    m_descriptorSetTwo = InsertionSortUtil::createDescriptorSet(
        logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        m_valueAndIndexBuffer,
        m_wasSwappedBuffer,
        m_dataSizeBuffer,
        m_offsetTwoBuffer,
        numberOfElements);

    m_commandBuffer = InsertionSortUtil::createCommandBuffer(
        logicalDevice,
        commandPool,
        m_pipeline,
        m_pipelineLayout,
        m_descriptorSetOne,
        m_descriptorSetTwo,
        m_valueAndIndexBuffer,
        m_wasSwappedBuffer,
        m_wasSwappedBufferHostVisible,
        numberOfElements);

    m_setDataSizeCommandBuffer = Buffer::recordCopyCommand(
        logicalDevice,
        commandPool,
        m_dataSizeBufferHostVisible,
        m_dataSizeBuffer,
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

InsertionSort::~InsertionSort() {
    vkFreeMemory(m_logicalDevice, m_valueAndIndexBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_valueAndIndexBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_wasSwappedBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_wasSwappedBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_wasSwappedBufferMemoryHostVisible, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_wasSwappedBufferHostVisible, nullptr);

    vkFreeMemory(m_logicalDevice, m_dataSizeBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_dataSizeBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_dataSizeBufferMemoryHostVisible, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_dataSizeBufferHostVisible, nullptr);

    vkFreeMemory(m_logicalDevice, m_offsetOneBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_offsetOneBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_offsetTwoBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_offsetTwoBuffer, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_descriptorSetLayout, nullptr);

    std::array<VkCommandBuffer, 2> commandBuffers = {m_commandBuffer, m_setDataSizeCommandBuffer};
    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, commandBuffers.size(), commandBuffers.data());

    vkDestroyDescriptorPool(m_logicalDevice, m_descriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_pipeline, nullptr);

    vkDestroyFence(m_logicalDevice, m_fence, nullptr);
}

void InsertionSort::setDataSize(uint32_t dataSize) {

    if (m_currentDataSize == dataSize) {
        return;
    }

    void* dataMap;
    vkMapMemory(m_logicalDevice, m_dataSizeBufferMemoryHostVisible, 0, sizeof(uint32_t), 0, &dataMap);
    uint32_t dataSizeCopy = dataSize;
    memcpy(dataMap, &dataSizeCopy, sizeof(uint32_t));
    vkUnmapMemory(m_logicalDevice, m_dataSizeBufferMemoryHostVisible);

    VkSubmitInfo submitInfoOne{};
    submitInfoOne.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfoOne.commandBufferCount = 1;
    submitInfoOne.pCommandBuffers = &m_setDataSizeCommandBuffer;

    vkResetFences(m_logicalDevice, 1, &m_fence);

    if (vkQueueSubmit(m_queue, 1, &submitInfoOne, m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit insertion sort set data size command buffer");
    }
    vkWaitForFences(m_logicalDevice, 1, &m_fence, VK_TRUE, UINT64_MAX);

    m_currentDataSize = dataSize;
}

void InsertionSort::setWasSwappedToZero() {
    void* dataMap;
    vkMapMemory(m_logicalDevice, m_wasSwappedBufferMemoryHostVisible, 0, sizeof(uint32_t), 0, &dataMap);
    uint32_t zero = 0;
    memcpy(dataMap, &zero, sizeof(uint32_t));
    vkUnmapMemory(m_logicalDevice, m_wasSwappedBufferMemoryHostVisible);
}

void InsertionSort::runSortCommands() {
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

uint32_t InsertionSort::needsSorting() {
    void* dataMap;
    vkMapMemory(m_logicalDevice, m_wasSwappedBufferMemoryHostVisible, 0, sizeof(uint32_t), 0, &dataMap);
    uint32_t wasSwappedValue = 0;
    memcpy(&wasSwappedValue, dataMap, sizeof(uint32_t));
    vkUnmapMemory(m_logicalDevice, m_wasSwappedBufferMemoryHostVisible);

    return wasSwappedValue;
}

void InsertionSort::run(uint32_t dataSize) {

    int numIterations = 0;

    {
        Timer time("Insertion Sort Vulkan");

        setDataSize(dataSize);

        do {
            setWasSwappedToZero();

            runSortCommands();

            numIterations += 1;
        } while (needsSorting());
    }

    std::cout << "Insertion Sort Vulkan total number of iterations = " << numIterations << "\n";
}
