#include <Renderer/InsertionSort.h>

#include <Renderer/Buffer.h>
#include <Renderer/Utils.h>
#include <Renderer/MyMath.h>
#include <Timer.h>

#include <vector>
#include <stdexcept>
#include <iostream>

#define NUMBER_OF_ELEMENTS X_DIM * 32

std::vector<InsertionSortUtil::ValueAndIndex> getData() {

    uint32_t numberOfElements = NUMBER_OF_ELEMENTS;
    std::vector<InsertionSortUtil::ValueAndIndex> data(numberOfElements);
    for (uint32_t i = 0; i < numberOfElements; ++i) {
        //data[i] = InsertionSortUtil::ValueAndIndex{MyMath::randomFloatBetweenZeroAndOne() * 100.0f, i};
        data[i] = InsertionSortUtil::ValueAndIndex{(NUMBER_OF_ELEMENTS * 2.0f) - i, i};
    }

    return data;
}

InsertionSort::InsertionSort(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool) {

    uint32_t numberOfElements = NUMBER_OF_ELEMENTS;

    m_serialData = getData();

    m_physicalDevice = physicalDevice;
    m_logicalDevice = logicalDevice;
    m_queue = queue;
    m_commandPool = commandPool;

    Buffer::createReadOnlyBuffer(
        m_serialData.data(),
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

    //std::cout << "wasSwappedValue = " << wasSwappedValue << "\n";
    return wasSwappedValue;
}

void insertionSortSerial(std::vector<InsertionSortUtil::ValueAndIndex>& data) {
    Timer time("Insertion Sort Serial");
    for (int i = 1; i < data.size(); ++i) {
        for (int j = i; j >= 1; --j) {
            InsertionSortUtil::ValueAndIndex left = data[j - 1];
            InsertionSortUtil::ValueAndIndex right = data[j];

            if (left.value > right.value) {
                data[j - 1] = right;
                data[j] = left;
            } else {
                break;
            }
        }
    }
}

void InsertionSort::printResults() {

    std::vector<InsertionSortUtil::ValueAndIndex> data(NUMBER_OF_ELEMENTS);

    Buffer::copyDeviceBufferToHost(
        data.data(),
        NUMBER_OF_ELEMENTS * sizeof(InsertionSortUtil::ValueAndIndex),
        m_valueAndIndexBuffer,
        m_physicalDevice,
        m_logicalDevice,
        m_commandPool,
        m_queue);

    int numErrors = 0;

    for (size_t i = 0; i < NUMBER_OF_ELEMENTS; ++i) {
        auto valueAndIndex = data[i];
        auto valueAndIndexSerial = m_serialData[i];
        //std::cout << "Value = " << valueAndIndex.value << " Index = " << valueAndIndex.index << "\n";

        if ((valueAndIndex.value != valueAndIndexSerial.value) || (valueAndIndex.index != valueAndIndexSerial.index)) {
            std::cout << "Mismatch at index = " << i << " GPU  = " << valueAndIndex.value << ", " << valueAndIndex.index
                << " SERIAL = " << valueAndIndexSerial.value << ", " << valueAndIndexSerial.index << "\n";
            numErrors += 1;
        }
    }

    std::cout << "Number of errors = " << numErrors << "\n";
}

void InsertionSort::runHelper() {

    int numIterations = 0;

    {
        Timer time("Insertion Sort GPU");
        do {
            setWasSwappedToZero();

            runSortCommands();

            numIterations += 1;
            //std::cout << "Insertion sort iteration number = " << numIterations << "\n";

        } while (needsSorting());
    }

    insertionSortSerial(m_serialData);

    InsertionSort::printResults();

    std::cout << "Insertion sort total number of iterations = " << numIterations << "\n";
}

void InsertionSort::run() {

    runHelper();
    runHelper();
    runHelper();
    runHelper();
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
