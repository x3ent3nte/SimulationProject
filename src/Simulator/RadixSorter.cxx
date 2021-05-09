#include <Simulator/RadixSorter.h>

#include <Utils/Buffer.h>
#include <Utils/Compute.h>

#include <array>
#include <stdexcept>

namespace {
    constexpr uint32_t kRadix = 2;
    constexpr uint32_t kNumberOfBits = sizeof(uint32_t) * 8;
} // namespace anonymous

namespace RadixSorterUtil {
    constexpr size_t kXDim = 512;
    constexpr size_t kRadixMapNumberOfBindings = 4;
    constexpr size_t kRadixScatterNumberOfBindings = 5;

    VkDescriptorSet createMapDescriptorSet(
        VkDevice logicalDevice,
        VkDescriptorSetLayout descriptorSetLayout,
        VkDescriptorPool descriptorPool,
        VkBuffer dataInBuffer,
        VkBuffer dataOutBuffer,
        VkBuffer radixBuffer,
        VkBuffer numberOfElementsBuffer,
        uint32_t maxNumberOfElements) {

        const size_t dataSize = maxNumberOfElements * sizeof(RadixSorter::ValueAndIndex);
        std::vector<Compute::BufferAndSize> bufferAndSizes = {
            {dataInBuffer, dataSize},
            {dataOutBuffer, dataSize},
            {radixBuffer, sizeof(uint32_t)},
            {numberOfElementsBuffer, sizeof(uint32_t)}
        };

        return Compute::createDescriptorSet(
            logicalDevice,
            descriptorSetLayout,
            descriptorPool,
            bufferAndSizes);
    }

    VkDescriptorSet createScatterDescriptorSet(
        VkDevice logicalDevice,
        VkDescriptorSetLayout descriptorSetLayout,
        VkDescriptorPool descriptorPool,
        VkBuffer dataInBuffer,
        VkBuffer scannedBuffer,
        VkBuffer dataOutBuffer,
        VkBuffer radixBuffer,
        VkBuffer numberOfElementsBuffer,
        uint32_t maxNumberOfElements) {

        const size_t dataSize = maxNumberOfElements * sizeof(RadixSorter::ValueAndIndex);
        std::vector<Compute::BufferAndSize> bufferAndSizes = {
            {dataInBuffer, dataSize},
            {scannedBuffer, maxNumberOfElements * sizeof(glm::uvec4)},
            {dataOutBuffer, dataSize},
            {radixBuffer, sizeof(uint32_t)},
            {numberOfElementsBuffer, sizeof(uint32_t)}
        };

        return Compute::createDescriptorSet(
            logicalDevice,
            descriptorSetLayout,
            descriptorPool,
            bufferAndSizes);
    }
}

RadixSorter::RadixSorter(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t maxNumberOfElements) {

    m_logicalDevice = logicalDevice;
    m_queue = queue;
    m_commandPool = commandPool;

    m_scanner = std::make_shared<Scanner<glm::uvec4>>(
        physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        maxNumberOfElements);

    // create buffers
    const size_t dataMemorySize = maxNumberOfElements * sizeof(ValueAndIndex);
    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        dataMemorySize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_dataBuffer,
        m_dataDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        dataMemorySize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_otherBuffer,
        m_otherDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_radixBuffer,
        m_radixDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_radixHostVisibleBuffer,
        m_radixHostVisibleDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_numberOfElementsBuffer,
        m_numberOfElementsDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_numberOfElementsHostVisibleBuffer,
        m_numberOfElementsHostVisibleDeviceMemory);

    // create pipeline
    // map pipeline
    m_mapDescriptorSetLayout = Compute::createDescriptorSetLayout(m_logicalDevice, RadixSorterUtil::kRadixMapNumberOfBindings);
    m_mapDescriptorPool = Compute::createDescriptorPool(m_logicalDevice, RadixSorterUtil::kRadixMapNumberOfBindings, 2);
    m_mapPipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_mapDescriptorSetLayout);
    m_mapPipeline = Compute::createPipeline("src/GLSL/spv/RadixMap.spv", m_logicalDevice, m_mapPipelineLayout);

    m_mapDescriptorSetOne = RadixSorterUtil::createMapDescriptorSet(
        m_logicalDevice,
        m_mapDescriptorSetLayout,
        m_mapDescriptorPool,
        m_dataBuffer,
        m_otherBuffer,
        m_radixBuffer,
        m_numberOfElementsBuffer,
        maxNumberOfElements);

    m_mapDescriptorSetTwo = RadixSorterUtil::createMapDescriptorSet(
        m_logicalDevice,
        m_mapDescriptorSetLayout,
        m_mapDescriptorPool,
        m_otherBuffer,
        m_dataBuffer,
        m_radixBuffer,
        m_numberOfElementsBuffer,
        maxNumberOfElements);

    // scatter pipeline
    m_scatterDescriptorSetLayout = Compute::createDescriptorSetLayout(m_logicalDevice, RadixSorterUtil::kRadixScatterNumberOfBindings);
    m_scatterDescriptorPool = Compute::createDescriptorPool(m_logicalDevice, RadixSorterUtil::kRadixScatterNumberOfBindings, 2);
    m_scatterPipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_scatterDescriptorSetLayout);
    m_scatterPipeline = Compute::createPipeline("src/GLSL/spv/RadixScatter.spv", m_logicalDevice, m_scatterPipelineLayout);

    m_scatterDescriptorSetOne = RadixSorterUtil::createScatterDescriptorSet(
        m_logicalDevice,
        m_scatterDescriptorSetLayout,
        m_scatterDescriptorPool,
        m_dataBuffer,
        m_scanner->m_dataBuffer,
        m_otherBuffer,
        m_radixBuffer,
        m_numberOfElementsBuffer,
        maxNumberOfElements);

    m_scatterDescriptorSetTwo = RadixSorterUtil::createScatterDescriptorSet(
        m_logicalDevice,
        m_scatterDescriptorSetLayout,
        m_scatterDescriptorPool,
        m_otherBuffer,
        m_scanner->m_dataBuffer,
        m_dataBuffer,
        m_radixBuffer,
        m_numberOfElementsBuffer,
        maxNumberOfElements);

    // create commands
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute fence");
    }

    m_copyBuffersCommandBuffer = Buffer::recordCopyCommand(
        m_logicalDevice,
        m_commandPool,
        m_otherBuffer,
        m_dataBuffer,
        dataMemorySize);

    m_setNumberOfElementsCommandBuffer = Buffer::recordCopyCommand(
        m_logicalDevice,
        m_commandPool,
        m_numberOfElementsHostVisibleBuffer,
        m_numberOfElementsBuffer,
        sizeof(uint32_t));

    setNumberOfElements(maxNumberOfElements);
    createCommandBuffers();
}

RadixSorter::~RadixSorter() {
    // free commands
    std::array<VkCommandBuffer, 2> commandBuffers = {
        m_setNumberOfElementsCommandBuffer,
        m_copyBuffersCommandBuffer};
    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, commandBuffers.size(), commandBuffers.data());
    destroyCommandBuffers();

    // free buffers
    vkFreeMemory(m_logicalDevice, m_dataDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_dataBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_otherDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_otherBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_radixDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_radixBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_radixHostVisibleDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_radixHostVisibleBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfElementsDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfElementsBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_numberOfElementsHostVisibleDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_numberOfElementsHostVisibleBuffer, nullptr);

    // free pipeline
    vkDestroyDescriptorSetLayout(m_logicalDevice, m_mapDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_logicalDevice, m_mapDescriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_mapPipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_mapPipeline, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_scatterDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_logicalDevice, m_scatterDescriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_scatterPipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_scatterPipeline, nullptr);

    vkDestroyFence(m_logicalDevice, m_fence, nullptr);
}

void RadixSorter::destroyCommandBuffers() {

}

void RadixSorter::createCommandBuffers() {

}

void RadixSorter::setNumberOfElements(uint32_t numberOfElements) {
    m_currentNumberOfElements = numberOfElements;

    void* dataMap;
    vkMapMemory(m_logicalDevice, m_numberOfElementsHostVisibleDeviceMemory, 0, sizeof(uint32_t), 0, &dataMap);
    uint32_t numberOfElementsCopy = numberOfElements;
    memcpy(dataMap, &numberOfElementsCopy, sizeof(uint32_t));
    vkUnmapMemory(m_logicalDevice, m_numberOfElementsHostVisibleDeviceMemory);

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

void RadixSorter::createCommandBuffersIfNecessary(uint32_t numberOfElements) {
    if (numberOfElements != m_currentNumberOfElements) {
        setNumberOfElements(numberOfElements);
        destroyCommandBuffers();
        createCommandBuffers();
    }
}

void RadixSorter::copyBuffers() {
    VkSubmitInfo submitInfoOne{};
    submitInfoOne.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfoOne.commandBufferCount = 1;
    submitInfoOne.pCommandBuffers = &m_copyBuffersCommandBuffer;

    vkResetFences(m_logicalDevice, 1, &m_fence);

    if (vkQueueSubmit(m_queue, 1, &submitInfoOne, m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit insertion sort set data size command buffer");
    }
    vkWaitForFences(m_logicalDevice, 1, &m_fence, VK_TRUE, UINT64_MAX);
}

void RadixSorter::setRadix(uint32_t radix) {
    void* dataMap;
    vkMapMemory(m_logicalDevice, m_radixHostVisibleDeviceMemory, 0, sizeof(uint32_t), 0, &dataMap);
    uint32_t radixCopy = radix;
    memcpy(dataMap, &radixCopy, sizeof(uint32_t));
    vkUnmapMemory(m_logicalDevice, m_radixHostVisibleDeviceMemory);
}

bool RadixSorter::needsSorting() {
    return true;
}

void RadixSorter::mapRadixToUVec4() {

}

void RadixSorter::scatter() {

}

void RadixSorter::sortAtRadix(uint32_t radix) {
    setRadix(radix);
    mapRadixToUVec4();
    m_scanner->run(m_currentNumberOfElements);
    scatter();
}

void RadixSorter::sort() {
    bool needsCopyAfterwards = false;

    for (uint32_t radix = 0; radix < kNumberOfBits; radix += kRadix) {
        if (needsSorting()) {
            sortAtRadix(radix);
            needsCopyAfterwards = !needsCopyAfterwards;
        } else {
            break;
        }
    }

    if (needsCopyAfterwards) {
        copyBuffers();
    }
}

void RadixSorter::run(uint32_t numberOfElements) {
    createCommandBuffersIfNecessary(numberOfElements);
    sort();
}
