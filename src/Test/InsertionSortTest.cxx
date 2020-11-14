#include <Test/InsertionSortTest.h>

#include <Test/TestUtils.h>
#include <Utils/MyMath.h>

#include <algorithm>
#include <iostream>

#define MAX_NUMBER_OF_ELEMENTS 64 * 1024

namespace {

    std::vector<float> serialSort(const std::vector<float>& data) {
        std::vector<float> sorted(data);

        std::sort(sorted.begin(), sorted.end());
        return sorted;
    }

    void expectEqual(const std::vector<float>& expected, const std::vector<float>& actual) {
        TestUtils::assertEqual(expected.size(), actual.size());

        int numberOfErrors = 0;

        for (int i = 0; i < expected.size(); ++i) {
            if (expected[i] != actual[i]) {
                numberOfErrors += 1;

                std::cout << "Mismatch at index = " << i << " Expected = " << expected[i] << " Actual = " << actual[i] << "\n";
            }
        }

        TestUtils::assertTrue(numberOfErrors == 0);
    }

    void testHelper(const std::vector<float>& data, std::shared_ptr<InsertionSortVulkanTest> vulkanTest) {
        std::cout << "Number of elements = " << data.size() << "\n";

        auto expected = serialSort(data);

        auto actualVulkan = vulkanTest->run(data);
        expectEqual(expected, actualVulkan);

        auto actualVulkanTwo = vulkanTest->run(actualVulkan);
        expectEqual(expected, actualVulkanTwo);

        auto actualVulkanThree = vulkanTest->run(actualVulkanTwo);
        expectEqual(expected, actualVulkanThree);

        auto actualCuda = InsertionSortCudaTest::run(data);
        expectEqual(expected, actualCuda);

        auto actualCudaTwo = InsertionSortCudaTest::run(actualCuda);
        expectEqual(expected, actualCudaTwo);

        auto actualCudaThree = InsertionSortCudaTest::run(actualCudaTwo);
        expectEqual(expected, actualCudaThree);
    }

    std::vector<float> generateDataWithReverseOrder(uint32_t size) {
        std::vector<float> data(size);
        for (uint32_t i = 0; i < size; ++i) {
            data[i] = (size - 1.0f) - i;
        }
        return data;
    }

    void testReverseOrder(
        std::shared_ptr<InsertionSortVulkanTest> vulkanTest,
        const std::vector<uint32_t>& sizes) {

        for (int i = 0; i < sizes.size(); ++i) {
            testHelper(generateDataWithReverseOrder(sizes[i]), vulkanTest);
        }
    }

    std::vector<float> generateDataWithRepeatedOrder(uint32_t size) {
        std::vector<float> data(size);
        for (uint32_t i = 0; i < size; ++i) {
            data[i] = -1.23;
        }
        return data;
    }

    void testRepeatedOrder(
        std::shared_ptr<InsertionSortVulkanTest> vulkanTest,
        const std::vector<uint32_t>& sizes) {

        for (int i = 0; i < sizes.size(); ++i) {
            testHelper(generateDataWithRepeatedOrder(sizes[i]), vulkanTest);
        }
    }

    std::vector<float> generateDataWithRandomOrder(uint32_t size) {
        std::vector<float> data(size);
        for (uint32_t i = 0; i < size; ++i) {
            data[i] = MyMath::randomFloatBetweenZeroAndOne() * 100.0f;
        }
        return data;
    }

    void testRandomOrder(
        std::shared_ptr<InsertionSortVulkanTest> vulkanTest,
        const std::vector<uint32_t>& sizes) {

        for (int i = 0; i < sizes.size(); ++i) {
            testHelper(generateDataWithRandomOrder(sizes[i]), vulkanTest);
        }
    }
} // end namespace anonymous

InsertionSortTest::InsertionSortTest(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool) {
    m_vulkanTest = std::make_shared<InsertionSortVulkanTest>(physicalDevice, logicalDevice, queue, commandPool, MAX_NUMBER_OF_ELEMENTS);
}

void InsertionSortTest::run() {

    std::vector<uint32_t> sizes = {MAX_NUMBER_OF_ELEMENTS, 1, 100, 99};

    TestUtils::testRunner("testReverseOrder", [this, sizes]() { testReverseOrder(m_vulkanTest, sizes); });
    TestUtils::testRunner("testRepeatedOrder", [this, sizes]() { testRepeatedOrder(m_vulkanTest, sizes); });
    //TestUtils::testRunner("testRandomOrder", [this, sizes]() { testRandomOrder(m_vulkanTest, sizes); });
}
