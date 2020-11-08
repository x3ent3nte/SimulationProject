#ifndef INSERTION_SORT_VULKAN_TEST_H
#define INSERTION_SORT_VULKAN_TEST_H

#include <Simulator/InsertionSort.h>

#include <vulkan/vulkan.h>

#include <vector>
#include <memory>

class InsertionSortVulkanTest {

private:

    VkPhysicalDevice m_physicalDevice;
    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    std::shared_ptr<InsertionSort> m_insertionSort;

public:
    InsertionSortVulkanTest(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        uint32_t numberOfElements);

    virtual ~InsertionSortVulkanTest();

    std::vector<float> run(const std::vector<float>& data);
};

#endif
