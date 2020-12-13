#include <Test/ReduceVulkanTest.h>

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

    m_reducer->run(data.size());
    return {0, 0, 1.0f};
}
