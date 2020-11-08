#include <Test/InsertionSortTest.h>

#include <Test/TestUtils.h>
#include <Utils/MyMath.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <functional>

#define NUMBER_OF_ELEMENTS X_DIM * 1024

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
        auto expected = serialSort(data);

        // When
        auto actual = vulkanTest->run(data);
        // Then
        expectEqual(expected, actual);

        // When
        auto actualTwo = vulkanTest->run(actual);
        // Then
        expectEqual(expected, actualTwo);
    }

    void testReverseOrder(std::shared_ptr<InsertionSortVulkanTest> vulkanTest) {

        size_t numberOfElements = NUMBER_OF_ELEMENTS;
        std::vector<float> data(numberOfElements);
        for (uint32_t i = 0; i < numberOfElements; ++i) {
            data[i] = (numberOfElements - 1.0f) - i;
        }

        testHelper(data, vulkanTest);
    }

    void testRepeatedOrder(std::shared_ptr<InsertionSortVulkanTest> vulkanTest) {

        size_t numberOfElements = NUMBER_OF_ELEMENTS;
        std::vector<float> data(numberOfElements);
        for (uint32_t i = 0; i < numberOfElements; ++i) {
            data[i] = -1.23;
        }

        testHelper(data, vulkanTest);
    }

    void testRandomOrder(std::shared_ptr<InsertionSortVulkanTest> vulkanTest) {
        size_t numberOfElements = NUMBER_OF_ELEMENTS;
        std::vector<float> data(numberOfElements);
        for (uint32_t i = 0; i < numberOfElements; ++i) {
            data[i] = MyMath::randomFloatBetweenZeroAndOne() * 100.0f;
        }

        testHelper(data, vulkanTest);
    }

    void testRunner(const std::string& name, std::function<void()> fn) {
        try {
            fn();
            std::cout << "\n[PASSED " << name << "]\n\n";
        } catch (const std::runtime_error& ex) {
            std::cout << "\n[FAILED " << name << "] " << ex.what() << "\n\n";
        }
    }
} // end namespace anonymous

InsertionSortTest::InsertionSortTest(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool) {
    m_vulkanTest = std::make_shared<InsertionSortVulkanTest>(physicalDevice, logicalDevice, queue, commandPool, NUMBER_OF_ELEMENTS);
}

void InsertionSortTest::run() {

    testRunner("testReverseOrder", [this]() { testReverseOrder(m_vulkanTest); });
    testRunner("testRepeatedOrder", [this]() { testRepeatedOrder(m_vulkanTest); });
    //testRunner("testRandomOrder", [this]() { testRandomOrder(m_vulkanTest); });
}
