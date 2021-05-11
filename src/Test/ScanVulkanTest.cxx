#include <Test/ScanVulkanTest.h>

#include <Utils/Buffer.h>
#include <Utils/Timer.h>

#include <Renderer/MyGLM.h>

template<typename T>
ScanVulkanTest<T>::ScanVulkanTest(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t numberOfElements)
    : m_physicalDevice(physicalDevice)
    , m_logicalDevice(logicalDevice)
    , m_queue(queue)
    , m_commandPool(commandPool)
    , m_scanner(std::make_shared<Scanner<T>>(
        m_physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        numberOfElements)) {}

template<typename T>
ScanVulkanTest<T>::~ScanVulkanTest() {

}

template<typename T>
std::vector<T> ScanVulkanTest<T>::run(const std::vector<T>& data) {

    std::vector<T> dataCopy = data;

    size_t bufferSize = dataCopy.size() * sizeof(T);

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

template class ScanVulkanTest<int32_t>;
template class ScanVulkanTest<glm::uvec4>;
