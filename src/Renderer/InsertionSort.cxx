#include <Renderer/InsertionSort.h>

#include <Renderer/Buffer.h>
#include <Renderer/Utils.h>
#include <Renderer/MyMath.h>

#include <vector>
#include <stdexcept>

InsertionSort::InsertionSort(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool) {

    uint32_t numberOfElements = 512 * 256;

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

    InsertionSortUtil::Info info{0, 0, numberOfElements};

    Buffer::createReadOnlyBuffer(
        &info,
        sizeof(InsertionSortUtil::Info),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        logicalDevice,
        commandPool,
        queue,
        m_infoBuffer,
        m_infoBufferMemory);

    m_descriptorSetLayout = InsertionSortUtil::createDescriptorSetLayout(logicalDevice);
    m_descriptorPool = InsertionSortUtil::createDescriptorPool(logicalDevice, 1);
    m_descriptorSet = InsertionSortUtil::createDescriptorSet(
        logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        m_valueAndIndexBuffer,
        m_infoBuffer,
        numberOfElements);

    m_pipelineLayout = InsertionSortUtil::createPipelineLayout(logicalDevice, m_descriptorSetLayout);

    auto shaderCode = Utils::readFile("src/GLSL/InsertionSort.spv");
    VkShaderModule shaderModule = Utils::createShaderModule(logicalDevice, shaderCode);

    m_pipeline = InsertionSortUtil::createPipeline(
        logicalDevice,
        shaderModule,
        m_descriptorSetLayout,
        m_pipelineLayout);

    m_commandBuffer = InsertionSortUtil::createCommandBuffer(
        logicalDevice,
        commandPool,
        m_pipeline,
        m_pipelineLayout,
        m_descriptorSet,
        numberOfElements);

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

    vkFreeMemory(logicalDevice, m_infoBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, m_infoBuffer, nullptr);

    vkDestroyDescriptorSetLayout(logicalDevice, m_descriptorSetLayout, nullptr);

    vkFreeCommandBuffers(logicalDevice, commandPool, 1, &m_commandBuffer);

    vkDestroyDescriptorPool(logicalDevice, m_descriptorPool, nullptr);
    vkDestroyPipelineLayout(logicalDevice, m_pipelineLayout, nullptr);
    vkDestroyPipeline(logicalDevice, m_pipeline, nullptr);

    vkDestroyFence(logicalDevice, m_fence, nullptr);
}
