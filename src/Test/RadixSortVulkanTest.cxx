#include <Test/RadixSortVulkanTest.h>

#include <Utils/Buffer.h>
#include <Utils/Timer.h>

RadixSortVulkanTest::RadixSortVulkanTest(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t maxNumberOfElements)
    : m_physicalDevice(physicalDevice)
    , m_logicalDevice(logicalDevice)
    , m_queue(queue)
    , m_commandPool(commandPool)
    , m_radixSorter(std::make_shared<RadixSorter>(physicalDevice, logicalDevice, queue, commandPool, maxNumberOfElements)) {

}

RadixSortVulkanTest::~RadixSortVulkanTest() {

}

std::vector<uint32_t> RadixSortVulkanTest::run(const std::vector<uint32_t>& numbers) {

    const size_t size = numbers.size();

    std::vector<RadixSorter::ValueAndIndex> valueAndIndexes(size);
    for (uint32_t i = 0; i < size; ++i) {
        valueAndIndexes[i] = {numbers[i], i};
    }

    const size_t bufferSize = size * sizeof(RadixSorter::ValueAndIndex);
    Buffer::copyHostToDeviceBuffer(
        valueAndIndexes.data(),
        bufferSize,
        m_radixSorter->m_dataBuffer,
        m_physicalDevice,
        m_logicalDevice,
        m_commandPool,
        m_queue);

    {
        Timer timer("Radix Sort Vulkan");
        m_radixSorter->run(size);
    }

    Buffer::copyDeviceBufferToHost(
        valueAndIndexes.data(),
        bufferSize,
        m_radixSorter->m_dataBuffer,
        m_physicalDevice,
        m_logicalDevice,
        m_commandPool,
        m_queue);

    std::vector<uint32_t> sorted(size);
    for (int i = 0; i < size; ++i) {
        sorted[i] = valueAndIndexes[i].value;
    }

    return sorted;
}
