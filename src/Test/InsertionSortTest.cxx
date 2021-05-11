#include <Test/InsertionSortTest.h>

#include <Utils/MyMath.h>
#include <Utils/Timer.h>
#include <Utils/TextColour.h>

#include <algorithm>
#include <sstream>
#include <iostream>

namespace {

    constexpr uint32_t kMaxNumberOfElements = 64 * 1024;

    const std::vector<uint32_t> kSizes = {
        kMaxNumberOfElements,
        kMaxNumberOfElements / 2,
        32 * 512,
        (32 * 512) + 1,
        1,
        2,
        100,
        99};

    std::vector<float> generateDataWithReverseOrder(uint32_t size) {
        std::vector<float> data(size);
        for (uint32_t i = 0; i < size; ++i) {
            data[i] = (size - 1.0f) - i;
        }
        return data;
    }

    std::vector<float> generateDataWithRepeatedOrder(uint32_t size) {
        std::vector<float> data(size);
        for (uint32_t i = 0; i < size; ++i) {
            data[i] = -1.23;
        }
        return data;
    }

    std::vector<float> generateDataWithRandomOrder(uint32_t size) {
        std::vector<float> data(size);
        for (uint32_t i = 0; i < size; ++i) {
            data[i] = MyMath::randomFloatBetweenZeroAndOne() * 100000.0f;
        }
        return data;
    }

    std::vector<float> serialSort(const std::vector<float>& data) {
        std::vector<float> sorted(data);

        {
            Timer timer("Std Sort Serial");
            std::sort(sorted.begin(), sorted.end());
        }
        return sorted;
    }

    std::vector<float> serialInsertionSort(const std::vector<float>& data) {
        std::vector<float> sorted(data);

        {
            Timer timer("Insertion Sort Serial");
            for (int i = 1; i < sorted.size(); ++i) {
                for (int j = i; j > 0; --j) {
                    float left = sorted[j - 1];
                    float right = sorted[j];
                    if (right < left) {
                        sorted[j - 1] = right;
                        sorted[j] = left;
                    } else {
                        break;
                    }
                }
            }
        }

        return sorted;
    }

    void testHelper(
        const std::vector<float>& data,
        std::shared_ptr<InsertionSortVulkanTest> vulkanTest,
        std::shared_ptr<TestInstance> testInstance) {

        std::cout << "Number of elements = " << data.size() << "\n";

        auto expected = serialSort(data);

        auto actualSerial = serialInsertionSort(expected);
        testInstance->assertEqual(expected, actualSerial);

        auto actualVulkan = vulkanTest->run(data);
        testInstance->assertEqual(expected, actualVulkan);

        auto actualVulkanTwo = vulkanTest->run(actualVulkan);
        testInstance->assertEqual(expected, actualVulkanTwo);

        auto actualVulkanThree = vulkanTest->run(actualVulkanTwo);
        testInstance->assertEqual(expected, actualVulkanThree);

        auto actualVulkanFour = vulkanTest->run(actualVulkanThree);
        testInstance->assertEqual(expected, actualVulkanFour);

        auto actualCuda = InsertionSortCudaTest::run(data);
        testInstance->assertEqual(expected, actualCuda);

        auto actualCudaTwo = InsertionSortCudaTest::run(actualCuda);
        testInstance->assertEqual(expected, actualCudaTwo);

        auto actualCudaThree = InsertionSortCudaTest::run(actualCudaTwo);
        testInstance->assertEqual(expected, actualCudaThree);

        auto actualCudaFour = InsertionSortCudaTest::run(actualCudaThree);
        testInstance->assertEqual(expected, actualCudaFour);
    }
} // end namespace anonymous

InsertionSortTest::InsertionSortTest(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool)
    : m_vulkanTest(std::make_shared<InsertionSortVulkanTest>(physicalDevice, logicalDevice, queue, commandPool, kMaxNumberOfElements)) {}

void InsertionSortTest::run(std::shared_ptr<TestRunner> testRunner) {

    std::cout << "\n" << TextColour::BLUE << "InsertionSortTest started " << TextColour::END << "\n";

    std::vector<std::pair<std::string, std::vector<float>(*)(uint32_t)>> nameAndFns = {
        {"Reverse", generateDataWithReverseOrder},
        {"Repeated", generateDataWithRepeatedOrder},
        {"Random", generateDataWithRandomOrder}
    };

    for (uint32_t size : kSizes) {
        for (const auto& nameAndFn : nameAndFns) {
            std::ostringstream testName;
            testName << "testInsertionSort_" << nameAndFn.first << "_" << size;
            auto fn = nameAndFn.second;
            testRunner->test(testName.str(), [this, fn, size](auto testInstance) {
                testHelper(fn(size), m_vulkanTest, testInstance);
            });
        }
    }

    std::cout << "\n" << TextColour::PURPLE << "InsertionSortTest finished" << TextColour::END << "\n";
}
