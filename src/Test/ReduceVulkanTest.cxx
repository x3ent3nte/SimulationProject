#include <Test/ReduceVulkanTest.h>

#include <Utils/Buffer.h>

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

ReducerUtil::Collision ReduceVulkanTest::run(const std::vector<ReducerUtil::Collision>& data) {

    size_t bufferSize = data.size() * sizeof(ReducerUtil::Collision);
    std::vector<ReducerUtil::Collision> dataCopy(data);

    Buffer::copyHostToDeviceBuffer(
        dataCopy.data(),
        bufferSize,
        m_reducer->m_oneBuffer,
        m_physicalDevice,
        m_logicalDevice,
        m_commandPool,
        m_queue);

    VkBuffer resultBuffer = m_reducer->run(data.size());

    ReducerUtil::Collision result;
    Buffer::copyDeviceBufferToHost(
        &result,
        sizeof(ReducerUtil::Collision),
        resultBuffer,
        m_physicalDevice,
        m_logicalDevice,
        m_commandPool,
        m_queue);

    return result;
}
