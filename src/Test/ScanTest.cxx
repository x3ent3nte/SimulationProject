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

    std::vector<int32_t> generateAllZeroes(uint32_t size) {
        std::vector<int32_t> data(size);
        for (int32_t i = 0; i < data.size(); ++i) {
            data[i] = 0;
        }
        return data;
    }

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

    std::vector<int32_t> generateIncreasing(uint32_t size) {
        std::vector<int32_t> data(size);
        for (int32_t i = 0; i < data.size(); ++i) {
            data[i] = i;
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
            data[i] = -i;
        }
        return data;
    }

    std::vector<int32_t> generateRandom(uint32_t size) {
        std::vector<int32_t> data(size);
        for (int32_t i = 0; i < data.size(); ++i) {
            data[i] = rand();;
        }
        return data;
    }

    std::vector<glm::uvec4> generateUVec4WithAllZeroes(uint32_t size) {
        std::vector<glm::uvec4> data(size);
        for (uint32_t i = 0; i < data.size(); ++i) {
            data[i] = {0, 0, 0, 0};
        }
        return data;
    }

    std::vector<glm::uvec4> generateUVec4WithAllOnes(uint32_t size) {
        std::vector<glm::uvec4> data(size);
        for (uint32_t i = 0; i < data.size(); ++i) {
            data[i] = {1, 1, 1, 1};
        }
        return data;
    }

    std::vector<glm::uvec4> generateUVec4AlternatingZeroAndOnes(uint32_t size) {
        std::vector<glm::uvec4> data(size);
        for (uint32_t i = 0; i < data.size(); ++i) {
            uint32_t mod = i % 2;
            uint32_t modPlusOne = (i + 1) % 2;
            data[i] = {mod, modPlusOne, mod, modPlusOne};
        }
        return data;
    }

    std::vector<glm::uvec4> generateUVec4Increasing(uint32_t size) {
        std::vector<glm::uvec4> data(size);
        for (uint32_t i = 0; i < data.size(); ++i) {
            data[i] = {i, i + 3, i + 1, i + 2};
        }
        return data;
    }

    std::vector<glm::uvec4> generateUVec4Decreasing(uint32_t size) {
        std::vector<glm::uvec4> data(size);
        uint32_t highest = size * 2;
        for (uint32_t i = 0; i < data.size(); ++i) {

            data[i] = {highest, highest - 2, highest - 1, highest - 3};
        }
        return data;
    }

    std::vector<glm::uvec4> generateUVec4Random(uint32_t size) {
        std::vector<glm::uvec4> data(size);
        for (uint32_t i = 0; i < data.size(); ++i) {
            data[i] = {rand(), rand(), rand(), rand()};
        }
        return data;
    }

    template<typename T>
    std::vector<T> serialScan(const std::vector<T>& data) {
        Timer timer("Scan Serial");
        std::vector<T> result(data.size());

        T count = data[0];
        result[0] = count;
        for (int32_t i = 1; i < data.size(); ++i) {
            count += data[i];
            result[i] = count;
        }

        return result;
    }

    template<typename T>
    void testHelper(
        const std::vector<T>& data,
        std::shared_ptr<ScanVulkanTest<T>> vulkanTest,
        std::shared_ptr<TestInstance> testInstance) {

        auto expected = serialScan<T>(data);

        auto actualVulkan = vulkanTest->run(data);
        testInstance->assertEqual(expected, actualVulkan);

        //auto actualCuda = ScanCudaTest::run(data);
        //testInstance->assertEqual(expected, actualCuda);
    }
} // namespace anonymous

ScanTest::ScanTest(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool)
    : m_vulkanTestInt32(std::make_shared<ScanVulkanTest<int32_t>>(
        physicalDevice,
        logicalDevice,
        queue,
        commandPool,
        kMaxNumberOfElements))
    , m_vulkanTestUVec4(std::make_shared<ScanVulkanTest<glm::uvec4>>(
        physicalDevice,
        logicalDevice,
        queue,
        commandPool,
        kMaxNumberOfElements)){

}

ScanTest::~ScanTest() {

}

void ScanTest::run(std::shared_ptr<TestRunner> testRunner) {
    std::cout << "\n" << TextColour::BLUE << "ScanTest started" << TextColour::END << "\n";

    std::vector<std::pair<std::string, std::vector<int32_t>(*)(uint32_t)>> nameAndFns = {
        {"AllZeroes", generateAllZeroes},
        {"AllOnes", generateAllOnes},
        {"AlternatingZeroAndOnes", generateAlternatingZeroAndOnes},
        {"Increasing", generateIncreasing},
        {"Decreasing", generateDecreasing},
        {"HasNegatives", generateHasNegatives},
        {"Random", generateRandom}
    };

    for (uint32_t size : kSizes) {
        for (const auto& nameAndFn : nameAndFns) {
            std::ostringstream testName;
            testName << "testScan_" << nameAndFn.first << "_" << size;
            auto fn = nameAndFn.second;
            testRunner->test(testName.str(), [this, fn, size](auto testInstance) {
                testHelper<int32_t>(fn(size), m_vulkanTestInt32, testInstance);
            });
        }
    }

    std::vector<std::pair<std::string, std::vector<glm::uvec4>(*)(uint32_t)>> vecNameAndFns = {
        {"AllZeroes", generateUVec4WithAllZeroes},
        {"AllOnes", generateUVec4WithAllOnes},
        {"AlternatingZeroAndOnes", generateUVec4AlternatingZeroAndOnes},
        {"Increasing", generateUVec4Increasing},
        {"Decreasing", generateUVec4Decreasing},
        {"Random", generateUVec4Random}
    };

    for (uint32_t size : kSizes) {
        for (const auto& nameAndFn : vecNameAndFns) {
            std::ostringstream testName;
            testName << "testScanUVec4_" << nameAndFn.first << "_" << size;
            auto fn = nameAndFn.second;
            testRunner->test(testName.str(), [this, fn, size](auto testInstance) {
                testHelper<glm::uvec4>(fn(size), m_vulkanTestUVec4, testInstance);
            });
        }
    }

    std::cout << "\n" << TextColour::PURPLE << "ScanTest finished" << TextColour::END << "\n";
}
