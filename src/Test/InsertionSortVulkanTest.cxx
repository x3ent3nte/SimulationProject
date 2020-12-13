#include <Test/InsertionSortVulkanTest.h>

#include <Utils/Buffer.h>

InsertionSortVulkanTest::InsertionSortVulkanTest(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t numberOfElements)
    : m_physicalDevice(physicalDevice)
    , m_logicalDevice(logicalDevice)
    , m_queue(queue)
    , m_commandPool(commandPool)
    , m_insertionSort(std::make_shared<InsertionSorter>(
        m_physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        numberOfElements)) {}

InsertionSortVulkanTest::~InsertionSortVulkanTest() {}

std::vector<float> InsertionSortVulkanTest::run(const std::vector<float>& data) {
    std::vector<InsertionSorterUtil::ValueAndIndex> valueAndIndexes(data.size());
    for (uint32_t i = 0; i < valueAndIndexes.size(); ++i) {
        valueAndIndexes[i] = {data[i], i};
    }

    size_t bufferSize = valueAndIndexes.size() * sizeof(InsertionSorterUtil::ValueAndIndex);

    Buffer::copyHostToDeviceBuffer(
        valueAndIndexes.data(),
        bufferSize,
        m_insertionSort->m_valueAndIndexBuffer,
        m_physicalDevice,
        m_logicalDevice,
        m_commandPool,
        m_queue);

    m_insertionSort->run(data.size());

    Buffer::copyDeviceBufferToHost(
        valueAndIndexes.data(),
        bufferSize,
        m_insertionSort->m_valueAndIndexBuffer,
        m_physicalDevice,
        m_logicalDevice,
        m_commandPool,
        m_queue);

    std::vector<float> sorted(valueAndIndexes.size());
    for (int i = 0; i < valueAndIndexes.size(); ++i) {
        sorted[i] = valueAndIndexes[i].value;
    }

    return sorted;
}
