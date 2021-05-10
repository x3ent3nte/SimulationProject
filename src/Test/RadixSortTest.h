#ifndef RADIX_SORT_TEST_H
#define RADIX_SORT_TEST_H

#include <Test/TestRunner.h>

#include <Test/RadixSortVulkanTest.h>

#include <vulkan/vulkan.h>

#include <memory>

class RadixSortTest {

private:

    const std::shared_ptr<RadixSortVulkanTest> m_vulkanTest;

public:

    RadixSortTest(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool);

    virtual ~RadixSortTest();

    void run(std::shared_ptr<TestRunner> testRunner);
};

#endif
