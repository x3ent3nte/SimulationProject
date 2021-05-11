#include <Test/ScanTest.h>

#include <Test/ScanCudaTest.cuh>
#include <Utils/Timer.h>
#include <Utils/TextColour.h>

#include <iostream>

namespace {

    constexpr uint32_t kMaxNumberOfElements = 1024 * 16 * 64;

    const std::vector<uint32_t> kSizes = {kMaxNumberOfElements, kMaxNumberOfElements / 2, 512, 1, 512 * 128, 2, 99, 100};

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
        std::vector<int> (*dataGenerator)(uint32_t),
        std::shared_ptr<TestInstance> testInstance) {

        for (uint32_t size : kSizes) {
            std::cout << "\nsize = " << size << "\n";
            testHelper(dataGenerator(size), vulkanTest, testInstance);
        }
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

void ScanTest::run(std::shared_ptr<TestRunner> testRunner) {
    std::cout << "\n" << TextColour::BLUE << "ScanTest started" << TextColour::END << "\n";

    testRunner->test("testAllOnes", [this](auto testInstance) {
        testDifferentSizesHelper(m_vulkanTest, generateAllOnes, testInstance);
    });

    testRunner->test("testAlternatingZeroAndOnes", [this](auto testInstance) {
        testDifferentSizesHelper(m_vulkanTest, generateAlternatingZeroAndOnes, testInstance);
    });

    testRunner->test("testDecreasing", [this](auto testInstance) {
        testDifferentSizesHelper(m_vulkanTest, generateAllOnes, testInstance);
    });

    testRunner->test("testHasNegatives", [this](auto testInstance) {
        testDifferentSizesHelper(m_vulkanTest, generateAlternatingZeroAndOnes, testInstance);
    });

    std::cout << "\n" << TextColour::PURPLE << "ScanTest finished" << TextColour::END << "\n";
}
