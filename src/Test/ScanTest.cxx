#include <Test/ScanTest.h>

#include <Test/ScanCudaTest.cuh>
#include <Utils/Timer.h>
#include <Utils/TextColour.h>

#include <sstream>
#include <iostream>

namespace {

    constexpr uint32_t kMaxNumberOfElements = 1024 * 16 * 64;

    const std::vector<uint32_t> kSizes = {
        kMaxNumberOfElements,
        kMaxNumberOfElements / 2,
        512,
        513,
        1,
        512 * 128,
        2,
        99,
        100};

    std::vector<int32_t> generateAllOnes(uint32_t size) {
        std::vector<int32_t> data(size);
        for (int32_t i = 0; i < data.size(); ++i) {
            data[i] = 1;
        }
        return data;
    }

    std::vector<int32_t> generateAlternatingZeroAndOnes(uint32_t size) {
        std::vector<int32_t> data(size);
        for (int32_t i = 0; i < data.size(); ++i) {
            data[i] = i % 2;
        }
        return data;
    }

    std::vector<int32_t> generateDecreasing(uint32_t size) {
        std::vector<int32_t> data(size);
        for (int32_t i = 0; i < data.size(); ++i) {
            data[i] = (size - i) - 1;
        }
        return data;
    }

    std::vector<int32_t> generateHasNegatives(uint32_t size) {
        std::vector<int32_t> data(size);
        for (int32_t i = 0; i < data.size(); ++i) {
            data[i] = - i;
        }
        return data;
    }

    std::vector<int32_t> serialScan(const std::vector<int32_t>& data) {
        Timer timer("Scan Serial");
        std::vector<int32_t> result(data.size());

        int32_t count = 0;
        for (int32_t i = 0; i < data.size(); ++i) {
            count += data[i];
            result[i] = count;
        }

        return result;
    }

    void testHelper(
        const std::vector<int32_t>& data,
        std::shared_ptr<ScanVulkanTest> vulkanTest,
        std::shared_ptr<TestInstance> testInstance) {

        auto expected = serialScan(data);

        auto actualVulkan = vulkanTest->run(data);
        testInstance->assertEqual(expected, actualVulkan);

        auto actualCuda = ScanCudaTest::run(data);
        testInstance->assertEqual(expected, actualCuda);
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

    std::vector<std::pair<std::string, std::vector<int32_t>(*)(uint32_t)>> nameAndFns = {
        {"AllOnes", generateAllOnes},
        {"AlternatingZeroAndOnes", generateAlternatingZeroAndOnes},
        {"Decreasing", generateDecreasing},
        {"HasNegatives", generateHasNegatives}
    };

    for (uint32_t size : kSizes) {
        for (const auto& nameAndFn : nameAndFns) {
            std::ostringstream testName;
            testName << "testScan_" << nameAndFn.first << "_" << size;
            auto fn = nameAndFn.second;
            testRunner->test(testName.str(), [this, fn, size](auto testInstance) {
                testHelper(fn(size), m_vulkanTest, testInstance);
            });
        }
    }

    std::cout << "\n" << TextColour::PURPLE << "ScanTest finished" << TextColour::END << "\n";
}
