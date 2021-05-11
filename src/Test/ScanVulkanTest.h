#ifndef SCAN_VULKAN_TEST_H
#define SCAN_VULKAN_TEST_H

#include <Simulator/Scanner.h>

#include <vulkan/vulkan.h>

#include <memory>
#include <vector>

template <typename T>
class ScanVulkanTest {

private:

    VkPhysicalDevice m_physicalDevice;
    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    std::shared_ptr<Scanner<T>> m_scanner;

public:

    ScanVulkanTest(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        uint32_t numberOfElements);

    virtual ~ScanVulkanTest();

    std::vector<T> run(const std::vector<T>& data);
};

#endif
