#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

#include <Test/InsertionSortTest.h>
#include <Test/ReduceTest.h>

#include <vulkan/vulkan.h>

class TestRunner {

private:

    VkPhysicalDevice m_physicalDevice;
    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

public:

    TestRunner(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool);

    virtual ~TestRunner() = default;

    void run();

};

#endif
