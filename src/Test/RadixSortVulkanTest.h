#ifndef RADIX_SORT_VULKAN_TEST_H
#define RADIX_SORT_VULKAN_TEST_H

#include <Simulator/RadixSorter.h>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

class RadixSortVulkanTest {
private:

    const VkPhysicalDevice m_physicalDevice;
    const VkDevice m_logicalDevice;
    const VkQueue m_queue;
    const VkCommandPool m_commandPool;

    const std::shared_ptr<RadixSorter> m_radixSorter;

public:

    RadixSortVulkanTest(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        uint32_t maxNumberOfElements);

    virtual ~RadixSortVulkanTest();

    std::vector<uint32_t> run(const std::vector<uint32_t>& numbers);
};

#endif
