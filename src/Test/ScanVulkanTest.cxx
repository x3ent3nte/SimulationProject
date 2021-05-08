#include <Test/ScanVulkanTest.h>

#include <Utils/Buffer.h>
#include <Utils/Timer.h>

ScanVulkanTest::ScanVulkanTest(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t numberOfElements)
    : m_physicalDevice(physicalDevice)
    , m_logicalDevice(logicalDevice)
    , m_queue(queue)
    , m_commandPool(commandPool)
    , m_scanner(std::make_shared<Scanner<int32_t>>(
        m_physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        numberOfElements)) {}

ScanVulkanTest::~ScanVulkanTest() {

}

std::vector<int> ScanVulkanTest::run(const std::vector<int>& data) {

    std::vector<int> dataCopy = data;

    size_t bufferSize = dataCopy.size() * sizeof(int);

    Buffer::copyHostToDeviceBuffer(
        dataCopy.data(),
        bufferSize,
        m_scanner->m_dataBuffer,
        m_physicalDevice,
        m_logicalDevice,
        m_commandPool,
        m_queue);

    {
        Timer timer("Scan Vulkan");
        m_scanner->run(dataCopy.size());
    }

    Buffer::copyDeviceBufferToHost(
        dataCopy.data(),
        bufferSize,
        m_scanner->m_dataBuffer,
        m_physicalDevice,
        m_logicalDevice,
        m_commandPool,
        m_queue);

    return dataCopy;
}
