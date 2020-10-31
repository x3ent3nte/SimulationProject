#include <Renderer/InsertionSort.h>

#include <Renderer/Buffer.h>
#include <Renderer/Utils.h>
#include <Renderer/MyMath.h>

#include <vector>
#include <stdexcept>
#include <iostream>

#define X_DIM 512

#define NUMBER_OF_ELEMENTS X_DIM * 256

InsertionSort::InsertionSort(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool) {

    uint32_t numberOfElements = NUMBER_OF_ELEMENTS;

    std::vector<InsertionSortUtil::ValueAndIndex> data(numberOfElements);
    for (uint32_t i = 0; i < numberOfElements; ++i) {
        data[i] = InsertionSortUtil::ValueAndIndex{MyMath::randomFloatBetweenZeroAndOne() * 100.0f, i};
    }

    m_logicalDevice = logicalDevice;
    m_queue = queue;

    Buffer::createReadOnlyBuffer(
        data.data(),
        numberOfElements * sizeof(InsertionSortUtil::ValueAndIndex),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        logicalDevice,
        commandPool,
        queue,
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

    InsertionSortUtil::Info infoOne{0, numberOfElements};
    Buffer::createReadOnlyBuffer(
        &infoOne,
        sizeof(InsertionSortUtil::Info),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        logicalDevice,
        commandPool,
        queue,
        m_infoOneBuffer,
        m_infoOneBufferMemory);

    InsertionSortUtil::Info infoTwo{X_DIM / 2, numberOfElements};
    Buffer::createReadOnlyBuffer(
        &infoTwo,
        sizeof(InsertionSortUtil::Info),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        logicalDevice,
        commandPool,
        queue,
        m_infoTwoBuffer,
        m_infoTwoBufferMemory);

    m_descriptorSetLayout = InsertionSortUtil::createDescriptorSetLayout(logicalDevice);
    m_descriptorPool = InsertionSortUtil::createDescriptorPool(logicalDevice, 2);
    m_pipelineLayout = InsertionSortUtil::createPipelineLayout(logicalDevice, m_descriptorSetLayout);

    auto shaderCode = Utils::readFile("src/GLSL/InsertionSort.spv");
    VkShaderModule shaderModule = Utils::createShaderModule(logicalDevice, shaderCode);

    m_pipeline = InsertionSortUtil::createPipeline(
        logicalDevice,
        shaderModule,
        m_descriptorSetLayout,
        m_pipelineLayout);

    m_descriptorSetOne = InsertionSortUtil::createDescriptorSet(
        logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        m_valueAndIndexBuffer,
        m_wasSwappedBuffer,
        m_infoOneBuffer,
        numberOfElements);

    m_descriptorSetTwo = InsertionSortUtil::createDescriptorSet(
        logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        m_valueAndIndexBuffer,
        m_wasSwappedBuffer,
        m_infoTwoBuffer,
        numberOfElements);

    m_commandBufferOne = InsertionSortUtil::createCommandBuffer(
        logicalDevice,
        commandPool,
        m_pipeline,
        m_pipelineLayout,
        m_descriptorSetOne,
        numberOfElements);

    m_commandBufferTwo = InsertionSortUtil::createCommandBuffer(
        logicalDevice,
        commandPool,
        m_pipeline,
        m_pipelineLayout,
        m_descriptorSetTwo,
        numberOfElements);

    m_copyWasSwappedFromHostToDevice = Buffer::recordCopyCommand(
        logicalDevice,
        commandPool,
        m_wasSwappedBufferHostVisible,
        m_wasSwappedBuffer,
        sizeof(uint32_t));

    m_copyWasSwappedFromDeviceToHost = Buffer::recordCopyCommand(
        logicalDevice,
        commandPool,
        m_wasSwappedBuffer,
        m_wasSwappedBufferHostVisible,
        sizeof(uint32_t));

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    if (vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &m_semaphore) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create semaphore");
    }

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute fence");
    }

    vkDestroyShaderModule(logicalDevice, shaderModule, nullptr);
}

void InsertionSort::runCopyCommand(VkCommandBuffer commandBuffer) {
    vkResetFences(m_logicalDevice, 1, &m_fence);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    if (vkQueueSubmit(m_queue, 1, &submitInfo, m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit was swapped copy command buffer");
    }
    vkWaitForFences(m_logicalDevice, 1, &m_fence, VK_TRUE, UINT64_MAX);
}

void InsertionSort::setWasSwappedToZero() {
    void* dataMap;
    vkMapMemory(m_logicalDevice, m_wasSwappedBufferMemoryHostVisible, 0, sizeof(uint32_t), 0, &dataMap);
    uint32_t zero = 0;
    memcpy(dataMap, &zero, sizeof(uint32_t));
    vkUnmapMemory(m_logicalDevice, m_wasSwappedBufferMemoryHostVisible);

    runCopyCommand(m_copyWasSwappedFromHostToDevice);
}

void InsertionSort::runSortCommands() {
    VkSubmitInfo submitInfoOne{};
    submitInfoOne.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfoOne.commandBufferCount = 1;
    submitInfoOne.pCommandBuffers = &m_commandBufferOne;
    submitInfoOne.signalSemaphoreCount = 1;
    submitInfoOne.pSignalSemaphores = &m_semaphore;

    VkSubmitInfo submitInfoTwo{};
    submitInfoTwo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfoTwo.commandBufferCount = 1;
    submitInfoTwo.pCommandBuffers = &m_commandBufferTwo;
    submitInfoTwo.waitSemaphoreCount = 1;
    submitInfoTwo.pWaitSemaphores = &m_semaphore;

    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
    submitInfoTwo.pWaitDstStageMask = waitStages;

    std::array<VkSubmitInfo, 2> submits = {submitInfoOne, submitInfoTwo};

    vkResetFences(m_logicalDevice, 1, &m_fence);

    if (vkQueueSubmit(m_queue, submits.size(), submits.data(), m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit insertion sort command buffer");
    }
    vkWaitForFences(m_logicalDevice, 1, &m_fence, VK_TRUE, UINT64_MAX);
}

uint32_t InsertionSort::needsSorting() {
    runCopyCommand(m_copyWasSwappedFromDeviceToHost);

    void* dataMap;
    vkMapMemory(m_logicalDevice, m_wasSwappedBufferMemoryHostVisible, 0, sizeof(uint32_t), 0, &dataMap);
    uint32_t wasSwappedValue = 0;
    memcpy(&wasSwappedValue, dataMap, sizeof(uint32_t));
    vkUnmapMemory(m_logicalDevice, m_wasSwappedBufferMemoryHostVisible);

    std::cout << "wasSwappedValue = " << wasSwappedValue << "\n";
    return wasSwappedValue;
}

void InsertionSort::run() {
    uint32_t numberOfElements = NUMBER_OF_ELEMENTS;

    int numIterations = 0;

    do {
        setWasSwappedToZero();

        runSortCommands();

        numIterations += 1;
        std::cout << "Insertion sort iteration number = " << numIterations << "\n";

        if (numIterations > 100) {
            break;
        }
    } while (needsSorting());

    std::cout << "Insertion sort total number of iterations = " << numIterations << "\n";
}

void InsertionSort::cleanUp(VkDevice logicalDevice, VkCommandPool commandPool) {
    vkFreeMemory(logicalDevice, m_valueAndIndexBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, m_valueAndIndexBuffer, nullptr);

    vkFreeMemory(logicalDevice, m_wasSwappedBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, m_wasSwappedBuffer, nullptr);

    vkFreeMemory(logicalDevice, m_wasSwappedBufferMemoryHostVisible, nullptr);
    vkDestroyBuffer(logicalDevice,m_wasSwappedBufferHostVisible, nullptr);

    vkFreeMemory(logicalDevice, m_infoOneBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, m_infoOneBuffer, nullptr);

    vkFreeMemory(logicalDevice, m_infoTwoBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, m_infoTwoBuffer, nullptr);

    vkDestroyDescriptorSetLayout(logicalDevice, m_descriptorSetLayout, nullptr);

    vkFreeCommandBuffers(logicalDevice, commandPool, 1, &m_commandBufferOne);
    vkFreeCommandBuffers(logicalDevice, commandPool, 1, &m_commandBufferTwo);
    vkFreeCommandBuffers(logicalDevice, commandPool, 1, &m_copyWasSwappedFromHostToDevice);
    vkFreeCommandBuffers(logicalDevice, commandPool, 1, &m_copyWasSwappedFromDeviceToHost);

    vkDestroyDescriptorPool(logicalDevice, m_descriptorPool, nullptr);
    vkDestroyPipelineLayout(logicalDevice, m_pipelineLayout, nullptr);
    vkDestroyPipeline(logicalDevice, m_pipeline, nullptr);

    vkDestroySemaphore(logicalDevice, m_semaphore, nullptr);
    vkDestroyFence(logicalDevice, m_fence, nullptr);
}
