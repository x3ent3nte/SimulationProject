#include <Test/InsertionSortVulkanTest.h>

#include <Utils/Buffer.h>

InsertionSortVulkanTest::InsertionSortVulkanTest(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t numberOfElements) {

    m_physicalDevice = physicalDevice;
    m_logicalDevice = logicalDevice;
    m_queue = queue;
    m_commandPool = commandPool;

    m_insertionSort = std::make_shared<InsertionSort>(physicalDevice, logicalDevice, queue, commandPool, numberOfElements);
}

InsertionSortVulkanTest::~InsertionSortVulkanTest() {
    m_insertionSort->cleanUp(m_logicalDevice, m_commandPool);
}

std::vector<float> InsertionSortVulkanTest::run(const std::vector<float>& data) {
    std::vector<InsertionSortUtil::ValueAndIndex> valueAndIndexes(data.size());
    for (uint32_t i = 0; i < valueAndIndexes.size(); ++i) {
        valueAndIndexes[i] = {data[i], i};
    }

    size_t bufferSize = valueAndIndexes.size() * sizeof(InsertionSortUtil::ValueAndIndex);

    Buffer::copyHostToDeviceBuffer(
        valueAndIndexes.data(),
        bufferSize,
        m_insertionSort->m_valueAndIndexBuffer,
        m_physicalDevice,
        m_logicalDevice,
        m_commandPool,
        m_queue);

    m_insertionSort->run();

    Buffer::copyDeviceBufferToHost(
        valueAndIndexes.data(),
        bufferSize,
        m_insertionSort->m_valueAndIndexBuffer,
        m_physicalDevice,
        m_logicalDevice,
        m_commandPool,
        m_queue);

    std::vector<float> sorted;

    for (int i = 0; i < valueAndIndexes.size(); ++i) {
        sorted.push_back(valueAndIndexes[i].value);
    }

    return sorted;
}
