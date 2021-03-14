#ifndef INSERTION_SORT_TEST_H
#define INSERTION_SORT_TEST_H

#include <Test/InsertionSortCudaTest.cuh>
#include <Test/InsertionSortVulkanTest.h>
#include <Test/TestRunner.h>

#include <memory>

class InsertionSortTest {

private:

    std::shared_ptr<InsertionSortVulkanTest> m_vulkanTest;

public:

    InsertionSortTest(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool);

    virtual ~InsertionSortTest() = default;

    void run(std::shared_ptr<TestRunner> testRunner);
};

#endif
