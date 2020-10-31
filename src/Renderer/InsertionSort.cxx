#include <Renderer/InsertionSort.h>

#include <Renderer/Buffer.h>
#include <Renderer/Utils.h>
#include <Renderer/MyMath.h>

#include <vector>
#include <stdexcept>

#define X_DIM 512

InsertionSort::InsertionSort(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool) {

    uint32_t numberOfElements = X_DIM * 256;

    std::vector<InsertionSortUtil::ValueAndIndex> data(numberOfElements);
    for (uint32_t i = 0; i < numberOfElements; ++i) {
        data[i] = InsertionSortUtil::ValueAndIndex{MyMath::randomFloatBetweenZeroAndOne() * 100.0f, i};
    }

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

void InsertionSort::run() {

}

void InsertionSort::cleanUp(VkDevice logicalDevice, VkCommandPool commandPool) {
    vkFreeMemory(logicalDevice, m_valueAndIndexBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, m_valueAndIndexBuffer, nullptr);

    vkFreeMemory(logicalDevice, m_wasSwappedBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, m_wasSwappedBuffer, nullptr);

    vkFreeMemory(logicalDevice, m_infoOneBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, m_infoOneBuffer, nullptr);

    vkFreeMemory(logicalDevice, m_infoTwoBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, m_infoTwoBuffer, nullptr);

    vkDestroyDescriptorSetLayout(logicalDevice, m_descriptorSetLayout, nullptr);

    vkFreeCommandBuffers(logicalDevice, commandPool, 1, &m_commandBufferOne);
    vkFreeCommandBuffers(logicalDevice, commandPool, 1, &m_commandBufferTwo);

    vkDestroyDescriptorPool(logicalDevice, m_descriptorPool, nullptr);
    vkDestroyPipelineLayout(logicalDevice, m_pipelineLayout, nullptr);
    vkDestroyPipeline(logicalDevice, m_pipeline, nullptr);

    vkDestroySemaphore(logicalDevice, m_semaphore, nullptr);
    vkDestroyFence(logicalDevice, m_fence, nullptr);
}
