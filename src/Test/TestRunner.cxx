#include <Test/TestRunner.h>

TestRunner::TestRunner(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool)
    : m_physicalDevice(physicalDevice)
    , m_logicalDevice(logicalDevice)
    , m_queue(queue)
    , m_commandPool(commandPool) {}

void TestRunner::run() {
    {
        InsertionSortTest(m_physicalDevice, m_logicalDevice, m_queue, m_commandPool).run();
    }

    {
        ReduceTest().run();
    }
}
