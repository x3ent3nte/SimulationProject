#include <Simulator/Scanner.h>

#include <Utils/Buffer.h>
#include <Utils/Compute.h>

#include <vector>
#include <stdexcept>

namespace ScannerUtil {

constexpr size_t xDim = 512;
constexpr size_t kNumberOfBindings = 2;

struct Info {
    uint32_t dataOffset;
    uint32_t offsetOffset;
    uint32_t numberOfElements;
};

VkDescriptorSetLayout createDescriptorSetLayout(VkDevice logicalDevice) {
    return Compute::createDescriptorSetLayout(logicalDevice, kNumberOfBindings);
}

VkDescriptorPool createDescriptorPool(VkDevice logicalDevice, size_t maxSets) {
    return Compute::createDescriptorPool(logicalDevice, kNumberOfBindings, maxSets);
}

VkDescriptorSet createDescriptorSet(
    VkDevice logicalDevice,
    VkDescriptorSetLayout descriptorSetLayout,
    VkDescriptorPool descriptorPool,
    VkBuffer dataBuffer,
    VkBuffer infoBuffer,
    uint32_t numberOfElements) {

    std::vector<Compute::BufferAndSize> bufferAndSizes = {
        {dataBuffer, numberOfElements * sizeof(int32_t)},
        {infoBuffer, sizeof(Info)}
    };

    return Compute::createDescriptorSet(
        logicalDevice,
        descriptorSetLayout,
        descriptorPool,
        bufferAndSizes);
}

VkCommandBuffer createCommandBuffer(
    VkDevice logicalDevice,
    VkCommandPool commandPool) {

    VkCommandBuffer commandBuffer;

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(logicalDevice, &commandBufferAllocateInfo, &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute command buffer");
    }

    return commandBuffer;
}

} // namespace ScannerUtil

Scanner::Scanner(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t numberOfElements) {

    m_logicalDevice = logicalDevice;
    m_queue = queue;
    m_commandPool = commandPool;

    size_t bufferNumberOfElements = numberOfElements * 2;

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        bufferNumberOfElements * sizeof(int32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_dataBuffer,
        m_dataBufferMemory);

    ScannerUtil::Info info{0, numberOfElements, numberOfElements};

    Buffer::createBufferWithData(
        &info,
        sizeof(ScannerUtil::Info),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        logicalDevice,
        commandPool,
        queue,
        m_infoBuffer,
        m_infoBufferMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        sizeof(ScannerUtil::Info),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_infoBufferHostVisible,
        m_infoBufferMemoryHostVisible);

    m_descriptorSetLayout = ScannerUtil::createDescriptorSetLayout(m_logicalDevice);
    m_descriptorPool = ScannerUtil::createDescriptorPool(m_logicalDevice, 1);
    m_pipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_descriptorSetLayout);
    m_pipeline = Compute::createPipeline("src/GLSL/spv/Scan.spv", m_logicalDevice, m_pipelineLayout);

    m_descriptorSet = ScannerUtil::createDescriptorSet(
        m_logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        m_dataBuffer,
        m_infoBuffer,
        bufferNumberOfElements);

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute fence");
    }
}

Scanner::~Scanner() {
    vkFreeMemory(m_logicalDevice, m_dataBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_dataBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_infoBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_infoBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_infoBufferMemoryHostVisible, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_infoBufferHostVisible, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_descriptorSetLayout, nullptr);

    vkDestroyDescriptorPool(m_logicalDevice, m_descriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_pipeline, nullptr);

    vkDestroyFence(m_logicalDevice, m_fence, nullptr);
}

void Scanner::setNumberOfElements(uint32_t numberOfElements) {

}

void Scanner::runScanCommand(uint32_t numberOfElements) {

}

void Scanner::run(uint32_t numberOfElements) {

}
