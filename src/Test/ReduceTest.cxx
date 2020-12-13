#include <Test/ReduceTest.h>

namespace {
    constexpr uint32_t kMaxNumberOfElements = 64 * 1024;
} // end namespace anonymous

ReduceTest::ReduceTest(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool)
    : m_vulkanTest(std::make_shared<ReduceVulkanTest>(physicalDevice, logicalDevice, queue, commandPool, kMaxNumberOfElements)) {}

ReduceTest::~ReduceTest() {}

void ReduceTest::run() {

}
