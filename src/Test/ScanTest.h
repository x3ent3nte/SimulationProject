#ifndef SCAN_TEST_H
#define SCAN_TEST_H

#include <Test/ScanVulkanTest.h>
#include <Test/TestRunner.h>
#include <Utils/MyGLM.h>

#include <vulkan/vulkan.h>

#include <memory>
#include <vector>

class ScanTest {
private:

    const std::shared_ptr<ScanVulkanTest<int32_t>> m_vulkanTestInt32;
    const std::shared_ptr<ScanVulkanTest<glm::uvec4>> m_vulkanTestUVec4;

public:

    ScanTest(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool);

    virtual ~ScanTest();

    void run(std::shared_ptr<TestRunner> testRunner);
};

#endif
