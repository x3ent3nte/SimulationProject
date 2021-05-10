#include <Test/RadixSortVulkanTest.h>

RadixSortVulkanTest::RadixSortVulkanTest(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t maxNumberOfElements)
    : m_radixSorter(std::make_shared<RadixSorter>(physicalDevice, logicalDevice, queue, commandPool, maxNumberOfElements)) {

}

RadixSortVulkanTest::~RadixSortVulkanTest() {

}

std::vector<uint32_t> RadixSortVulkanTest::run(const std::vector<uint32_t>& numbers) {
    return numbers;
}
