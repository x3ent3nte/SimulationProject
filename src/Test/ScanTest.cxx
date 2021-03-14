#include <Test/ScanTest.h>

#include <Test/ScanCudaTest.cuh>
#include <Test/TestUtils.h>
#include <Utils/Timer.h>

#include <iostream>

namespace {

    constexpr uint32_t kMaxNumberOfElements = 1024 * 1024 * 64;

    std::vector<int> serialScan(const std::vector<int>& data) {
        Timer timer("Scan Serial");
        std::vector<int> result(data.size());

        int count = 0;
        for (int i = 0; i < data.size(); ++i) {
            count += data[i];
            result[i] = count;
        }

        return result;
    }

    std::vector<int> generateAllOnes(uint32_t size) {
        std::vector<int> data(size);
        for (int i = 0; i < data.size(); ++i) {
            data[i] = 1;
        }
        return data;
    }

    std::vector<int> generateAlternatingZeroAndOnes(uint32_t size) {
        std::vector<int> data(size);
        for (int i = 0; i < data.size(); ++i) {
            data[i] = i % 2;
        }
        return data;
    }

    std::vector<int> generateDecreasing(uint32_t size) {
        std::vector<int> data(size);
        for (int i = 0; i < data.size(); ++i) {
            data[i] = (size - i) - 1;
        }
        return data;
    }

    std::vector<int> generateHasNegatives(uint32_t size) {
        std::vector<int> data(size);
        for (int i = 0; i < data.size(); ++i) {
            data[i] = - i;
        }
        return data;
    }

    void testHelper(
        const std::vector<int>& data,
        std::shared_ptr<ScanVulkanTest> vulkanTest) {

        auto expected = serialScan(data);

        auto actualVulkan = vulkanTest->run(data);
        TestUtils::assertEqual(expected, actualVulkan);

        auto actualCuda = ScanCudaTest::run(data);
        TestUtils::assertEqual(expected, actualCuda);
    }

    void testDifferentSizesHelper(
        std::shared_ptr<ScanVulkanTest> vulkanTest,
        std::vector<uint32_t> sizes,
        std::vector<int> (*dataGenerator)(uint32_t)) {

        for (uint32_t size : sizes) {
            std::cout << "\nsize = " << size << "\n";
            auto data = dataGenerator(size);
            testHelper(data, vulkanTest);
        }
    }

    void testAllOnes(std::shared_ptr<ScanVulkanTest> vulkanTest, std::vector<uint32_t> sizes) {
        testDifferentSizesHelper(vulkanTest, sizes, generateAllOnes);
    }

    void testAlternatingZeroAndOnes(std::shared_ptr<ScanVulkanTest> vulkanTest, std::vector<uint32_t> sizes) {
        testDifferentSizesHelper(vulkanTest, sizes, generateAlternatingZeroAndOnes);
    }

    void testDecreasing(std::shared_ptr<ScanVulkanTest> vulkanTest, std::vector<uint32_t> sizes) {
        testDifferentSizesHelper(vulkanTest, sizes, generateDecreasing);
    }

    void testHasNegatives(std::shared_ptr<ScanVulkanTest> vulkanTest, std::vector<uint32_t> sizes) {
        testDifferentSizesHelper(vulkanTest, sizes, generateHasNegatives);
    }

} // namespace anonymous

ScanTest::ScanTest(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool)
    : m_vulkanTest(std::make_shared<ScanVulkanTest>(
        physicalDevice,
        logicalDevice,
        queue,
        commandPool,
        kMaxNumberOfElements)) {

}

ScanTest::~ScanTest() {

}

void ScanTest::run() {
    std::cout << "\n\033[94mScanTest started\033[0m\n";

    std::vector<uint32_t> sizes = {kMaxNumberOfElements, kMaxNumberOfElements / 2, 512, 1, 2, 99, 100};

    TestUtils::testRunner("testAllOnes", [this, &sizes]() { testAllOnes(m_vulkanTest, sizes); });
    TestUtils::testRunner("testAlternatingZeroAndOnes", [this, &sizes]() { testAlternatingZeroAndOnes(m_vulkanTest, sizes); });
    TestUtils::testRunner("testDecreasing", [this, &sizes]() { testDecreasing(m_vulkanTest, sizes); });
    TestUtils::testRunner("testHasNegatives", [this, &sizes]() { testHasNegatives(m_vulkanTest, sizes); });

    std::cout << "\n\033[95mScanTest finished\033[0m\n";
}
