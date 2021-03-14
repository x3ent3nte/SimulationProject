#include <Test/ScanTest.h>

#include <Test/ScanCudaTest.cuh>
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
        std::shared_ptr<ScanVulkanTest> vulkanTest,
        std::shared_ptr<TestInstance> testInstance) {

        auto expected = serialScan(data);

        auto actualVulkan = vulkanTest->run(data);
        testInstance->assertEqual(expected, actualVulkan);

        auto actualCuda = ScanCudaTest::run(data);
        testInstance->assertEqual(expected, actualCuda);
    }

    void testDifferentSizesHelper(
        std::shared_ptr<ScanVulkanTest> vulkanTest,
        std::vector<uint32_t> sizes,
        std::vector<int> (*dataGenerator)(uint32_t),
        std::shared_ptr<TestInstance> testInstance) {

        for (uint32_t size : sizes) {
            std::cout << "\nsize = " << size << "\n";
            auto data = dataGenerator(size);
            testHelper(data, vulkanTest, testInstance);
        }
    }

    void testAllOnes(
        std::shared_ptr<ScanVulkanTest> vulkanTest,
        std::vector<uint32_t> sizes,
        std::shared_ptr<TestInstance> testInstance) {

        testDifferentSizesHelper(vulkanTest, sizes, generateAllOnes, testInstance);
    }

    void testAlternatingZeroAndOnes(
        std::shared_ptr<ScanVulkanTest> vulkanTest,
        std::vector<uint32_t> sizes,
        std::shared_ptr<TestInstance> testInstance) {

        testDifferentSizesHelper(vulkanTest, sizes, generateAlternatingZeroAndOnes, testInstance);
    }

    void testDecreasing(
        std::shared_ptr<ScanVulkanTest> vulkanTest,
        std::vector<uint32_t> sizes,
        std::shared_ptr<TestInstance> testInstance) {

        testDifferentSizesHelper(vulkanTest, sizes, generateDecreasing, testInstance);
    }

    void testHasNegatives(
        std::shared_ptr<ScanVulkanTest> vulkanTest,
        std::vector<uint32_t> sizes,
        std::shared_ptr<TestInstance> testInstance) {
        testDifferentSizesHelper(vulkanTest, sizes, generateHasNegatives, testInstance);
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

void ScanTest::run(std::shared_ptr<TestInstance> testInstance) {
    std::cout << "\n\033[94mScanTest started\033[0m\n";

    std::vector<uint32_t> sizes = {kMaxNumberOfElements, kMaxNumberOfElements / 2, 512, 1, 2, 99, 100};

    testInstance->test("testAllOnes", [this, &sizes, testInstance]() { testAllOnes(m_vulkanTest, sizes, testInstance); });
    testInstance->test("testAlternatingZeroAndOnes", [this, &sizes, testInstance]() { testAlternatingZeroAndOnes(m_vulkanTest, sizes, testInstance); });
    testInstance->test("testDecreasing", [this, &sizes, testInstance]() { testDecreasing(m_vulkanTest, sizes, testInstance); });
    testInstance->test("testHasNegatives", [this, &sizes, testInstance]() { testHasNegatives(m_vulkanTest, sizes, testInstance); });

    std::cout << "\n\033[95mScanTest finished\033[0m\n";
}
