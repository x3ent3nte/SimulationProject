#ifndef REDUCE_TEST_H
#define REDUCE_TEST_H

#include <Test/ReduceVulkanTest.h>
#include <Test/ReduceCudaTest.cuh>

#include <vulkan/vulkan.h>

#include <memory>

class ReduceTest {
private:

    std::shared_ptr<ReduceVulkanTest> m_vulkanTest;

public:

    ReduceTest(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool);

    virtual ~ReduceTest();

    void run();
};

#endif
