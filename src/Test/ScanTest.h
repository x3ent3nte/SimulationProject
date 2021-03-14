#ifndef SCAN_TEST_H
#define SCAN_TEST_H

#include <Test/ScanVulkanTest.h>
#include <Test/TestInstance.h>

#include <vulkan/vulkan.h>

#include <memory>
#include <vector>

class ScanTest {
private:

    std::shared_ptr<ScanVulkanTest> m_vulkanTest;

public:

    ScanTest(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool);

    virtual ~ScanTest();

    void run(std::shared_ptr<TestInstance> testInstance);
};

#endif
