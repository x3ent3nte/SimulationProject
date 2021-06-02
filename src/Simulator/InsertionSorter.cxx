#include <Simulator/InsertionSorter.h>

#include <Simulator/Collision.h>
#include <Utils/Buffer.h>
#include <Utils/Utils.h>
#include <Utils/MyMath.h>
#include <Utils/Compute.h>
#include <Utils/Timer.h>
#include <Utils/Command.h>

#include <vector>
#include <stdexcept>
#include <iostream>

namespace InsertionSorterUtil {

constexpr size_t xDim = 256;

VkCommandBuffer createCommandBuffer(
    VkDevice logicalDevice,
    VkCommandPool commandPool,
    VkBuffer wasSwappedBuffer,
    VkBuffer wasSwappedBufferHostVisible,
    std::shared_ptr<ShaderLambda> lambdaOne,
    std::shared_ptr<ShaderLambda> lambdaTwo,
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
    copyRegion.size = sizeof(uint32_t);
    vkCmdCopyBuffer(commandBuffer, wasSwappedBufferHostVisible, wasSwappedBuffer, 1, &copyRegion);

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

    uint32_t xGroups = ceil(((float) numberOfElements) / ((float) 2 * xDim));
    //std::cout << "Number of X groups = " << xGroups << "\n";

    lambdaOne->record(commandBuffer, xGroups, 1, 1);

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

    lambdaTwo->record(commandBuffer, xGroups, 1, 1);

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

    vkCmdCopyBuffer(commandBuffer, wasSwappedBuffer, wasSwappedBufferHostVisible, 1, &copyRegion);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end compute command buffer");
    }

    return commandBuffer;
}

};

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
        numberOfElements * sizeof(ValueAndIndex),
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

    auto shaderFn = std::make_shared<ShaderFunction>(logicalDevice, 4, "src/GLSL/spv/InsertionSort.spv");
    auto shaderPool = std::make_shared<ShaderPool>(shaderFn, 2);

    std::vector<Compute::BufferAndSize> oneBufferAndSizes = {
        {m_valueAndIndexBuffer, numberOfElements * sizeof(ValueAndIndex)},
        {m_wasSwappedBuffer, sizeof(uint32_t)},
        {m_numberOfElementsBuffer, sizeof(uint32_t)},
        {m_offsetOneBuffer, sizeof(uint32_t)}
    };
    m_lambdaOne = std::make_shared<ShaderLambda>(shaderPool, oneBufferAndSizes);

    std::vector<Compute::BufferAndSize> twoBufferAndSizes = {
        {m_valueAndIndexBuffer, numberOfElements * sizeof(ValueAndIndex)},
        {m_wasSwappedBuffer, sizeof(uint32_t)},
        {m_numberOfElementsBuffer, sizeof(uint32_t)},
        {m_offsetTwoBuffer, sizeof(uint32_t)}
    };
    m_lambdaTwo = std::make_shared<ShaderLambda>(shaderPool, twoBufferAndSizes);

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

    Buffer::writeHostVisible(&numberOfElements, m_numberOfElementsBufferMemoryHostVisible, 0, sizeof(uint32_t), m_logicalDevice);

    Command::runAndWait(m_setNumberOfElementsCommandBuffer, m_fence, m_queue, m_logicalDevice);
}

void InsertionSorter::createCommandBuffer(uint32_t numberOfElements) {

    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &m_commandBuffer);

    m_commandBuffer = InsertionSorterUtil::createCommandBuffer(
        m_logicalDevice,
        m_commandPool,
        m_wasSwappedBuffer,
        m_wasSwappedBufferHostVisible,
        m_lambdaOne,
        m_lambdaTwo,
        numberOfElements);
}

void InsertionSorter::setWasSwappedToZero() {
    uint32_t zero = 0;
    Buffer::writeHostVisible(&zero, m_wasSwappedBufferMemoryHostVisible, 0, sizeof(uint32_t), m_logicalDevice);
}

uint32_t InsertionSorter::needsSorting() {
    uint32_t wasSwappedValue;
    Buffer::readHostVisible(m_wasSwappedBufferMemoryHostVisible, &wasSwappedValue, 0, sizeof(uint32_t), m_logicalDevice);
    return wasSwappedValue;
}

void InsertionSorter::run(uint32_t numberOfElements) {

    int numIterations = 0;

    {
        //Timer timer("Insertion Sort Vulkan");

        setNumberOfElements(numberOfElements);

        do {
            setWasSwappedToZero();

            Command::runAndWait(m_commandBuffer, m_fence, m_queue, m_logicalDevice);

            numIterations += 1;
        } while (needsSorting());
    }

    //std::cout << "Insertion Sort Vulkan total number of iterations = " << numIterations << "\n";
}
