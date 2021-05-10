#include <Test/TestApplication.h>

#include <Test/TestRunner.h>
#include <Test/InsertionSortTest.h>
#include <Test/ReduceTest.h>
#include <Test/ScanTest.h>
#include <Test/RadixSortTest.h>

#include <iostream>
#include <memory>

TestApplication::TestApplication(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool)
    : m_physicalDevice(physicalDevice)
    , m_logicalDevice(logicalDevice)
    , m_queue(queue)
    , m_commandPool(commandPool) {}

void TestApplication::run() {

    std::cout << "\n\033[94mTest Runner started\033[0m\n";

    auto testRunner = std::make_shared<TestRunner>();

    InsertionSortTest(m_physicalDevice, m_logicalDevice, m_queue, m_commandPool).run(testRunner);
    ReduceTest(m_physicalDevice, m_logicalDevice, m_queue, m_commandPool).run(testRunner);
    ScanTest(m_physicalDevice, m_logicalDevice, m_queue, m_commandPool).run(testRunner);
    RadixSortTest(m_physicalDevice, m_logicalDevice, m_queue, m_commandPool).run(testRunner);

    testRunner->report();

    std::cout << "\n\033[95mTest Runner finished\033[0m\n";
}
