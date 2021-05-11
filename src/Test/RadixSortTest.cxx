#include <Test/RadixSortTest.h>

#include <Test/RadixSortCudaTest.cuh>
#include <Utils/Timer.h>
#include <Utils/TextColour.h>

#include <algorithm>
#include <vector>
#include <sstream>
#include <stdlib.h>

namespace {

    constexpr uint32_t kMaxNumberOfElements = 1024 * 32;

    const std::vector<uint32_t> kSizes = {
        kMaxNumberOfElements,
        kMaxNumberOfElements / 2,
        512,
        1,
        512 * 16,
        (512 * 16) + 1,
        2,
        99,
        100};

    std::vector<uint32_t> generateReverse(uint32_t size) {
        std::cout << "Generating Reverse\n";
        std::vector<uint32_t> numbers(size);
        for (uint32_t i = 0; i < size; ++i) {
            numbers[i] = (size - 1) - i;
        }
        return numbers;
    }

    std::vector<uint32_t> generateRandom(uint32_t size) {
        std::cout << "Generating Random\n";
        std::vector<uint32_t> numbers(size);
        for (uint32_t i = 0; i < size; ++i) {
            numbers[i] = rand();
        }
        return numbers;
    }

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
        //const auto actualCuda = RadixSortCudaTest::run(numbers);

        testInstance->assertEqual(expected, actualVulkan);
        //testInstance->assertEqual(expected, actualCuda);
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
    std::cout << "\n" << TextColour::BLUE << "RadixSortTest started" << TextColour::END << "\n";

    std::vector<std::pair<std::string, std::vector<uint32_t>(*)(uint32_t)>> nameAndFns = {
        {"Reverse", generateReverse},
        {"Random", generateRandom}};

    for (uint32_t size : kSizes) {
        for (const auto& nameAndFn : nameAndFns) {

            std::ostringstream testName;
            testName << "testRadixSort_" << nameAndFn.first << "_" << size;
            auto fn = nameAndFn.second;
            testRunner->test(testName.str(), [this, fn, size](auto testInstance) {
                testHelper(fn(size), m_vulkanTest, testInstance);
            });
        }
    }

    std::cout << "\n" << TextColour::PURPLE << "RadixSortTest finished" << TextColour::END << "\n";
}
