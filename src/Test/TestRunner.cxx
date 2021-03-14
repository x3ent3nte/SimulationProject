#include <Test/TestRunner.h>

#include <Test/TestInstance.h>
#include <Test/InsertionSortTest.h>
#include <Test/ReduceTest.h>
#include <Test/ScanTest.h>

#include <iostream>
#include <memory>

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

    std::cout << "\n\033[94mTest Runner started\033[0m\n";

    auto testInstance = std::make_shared<TestInstance>();

    InsertionSortTest(m_physicalDevice, m_logicalDevice, m_queue, m_commandPool).run(testInstance);
    ReduceTest(m_physicalDevice, m_logicalDevice, m_queue, m_commandPool).run(testInstance);
    ScanTest(m_physicalDevice, m_logicalDevice, m_queue, m_commandPool).run(testInstance);

    testInstance->printReport();

    std::cout << "\n\033[95mTest Runner finished\033[0m\n";
}
