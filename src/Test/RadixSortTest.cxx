#include <Test/RadixSortTest.h>

#include <Test/RadixSortCudaTest.cuh>
#include <Utils/Timer.h>

#include <algorithm>
#include <vector>

namespace {

    constexpr uint32_t kMaxNumberOfElements = 1024 * 1024 * 16;

    std::vector<uint32_t> serialSort(const std::vector<uint32_t>& numbers) {
        std::vector<uint32_t> sorted(numbers);
        {
            Timer timer("Radix Std Sort Serial");
            std::sort(sorted.begin(), sorted.end());
        }
        return sorted;
    }

    void testHelper(
        const std::vector<uint32_t>& numbers,
        std::shared_ptr<RadixSortVulkanTest> vulkanTest,
        std::shared_ptr<TestInstance> testInstance) {

        const auto expected = serialSort(numbers);
        const auto actualVulkan = vulkanTest->run(numbers);
        const auto actualCuda = RadixSortCudaTest::run(numbers);

        testInstance->assertEqual(expected, actualVulkan);
        testInstance->assertEqual(expected, actualCuda);
    }
} // namespace anonymous

RadixSortTest::RadixSortTest(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool)
    : m_vulkanTest(std::make_shared<RadixSortVulkanTest>(physicalDevice, logicalDevice, queue, commandPool, kMaxNumberOfElements)) {}

RadixSortTest::~RadixSortTest() {

}

void RadixSortTest::run(std::shared_ptr<TestRunner> testRunner) {

}
