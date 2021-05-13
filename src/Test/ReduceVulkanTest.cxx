#include <Test/ReduceVulkanTest.h>

#include <Utils/Buffer.h>
#include <Utils/Timer.h>

ReduceVulkanTest::ReduceVulkanTest(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t numberOfElements)
    : m_physicalDevice(physicalDevice)
    , m_logicalDevice(logicalDevice)
    , m_queue(queue)
    , m_commandPool(commandPool)
    , m_reducer(std::make_shared<Reducer>(
        m_physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        numberOfElements)) {}

ReduceVulkanTest::~ReduceVulkanTest() {}

Collision ReduceVulkanTest::run(const std::vector<Collision>& data) {

    size_t bufferSize = data.size() * sizeof(Collision);
    std::vector<Collision> dataCopy(data);

    Buffer::copyHostToDeviceBuffer(
        dataCopy.data(),
        bufferSize,
        m_reducer->m_oneBuffer,
        m_physicalDevice,
        m_logicalDevice,
        m_commandPool,
        m_queue);

    VkBuffer resultBuffer;
    {
        Timer timer("Reduce Vulkan");
        resultBuffer = m_reducer->run(data.size());
    }

    Collision result;
    Buffer::copyDeviceBufferToHost(
        &result,
        sizeof(Collision),
        resultBuffer,
        m_physicalDevice,
        m_logicalDevice,
        m_commandPool,
        m_queue);

    return result;
}
