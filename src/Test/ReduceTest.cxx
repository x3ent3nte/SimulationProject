#include <Test/ReduceTest.h>

#include <Test/ReduceCudaTest.cuh>
#include <Simulator/Agent.h>
#include <Utils/Timer.h>
#include <Utils/TextColour.h>

#include <vector>
#include <iostream>
#include <sstream>

namespace {
    constexpr uint32_t kMaxNumberOfElements = 512 * 1024;

    const std::vector<uint32_t> kSizes = {
        kMaxNumberOfElements,
        kMaxNumberOfElements / 2,
        32 * 512,
        (32 * 512) + 1,
        1,
        2,
        100,
        99};

    Collision reduceSerial(const std::vector<Collision>& collisions) {
        auto earliest = collisions[0];

        {
            Timer timer("Reduce Serial");
            for (int i = 1; i < collisions.size(); ++i) {
                auto current = collisions[i];
                if (current.time < earliest.time) {
                    earliest = current;
                }
            }
        }

        return earliest;
    }

    std::vector<Collision> generateCollisions(uint32_t numberOfElements) {
        std::vector<Collision> collisions(numberOfElements);
        for (uint32_t i = 0; i < numberOfElements; ++i) {
            collisions[i] = {i, i + 1, float(numberOfElements - i)};
        }

        collisions[numberOfElements / 2] = {123, 321, 0};
        return collisions;
    }

    void printCollision(const Collision& collision, std::stringstream& ss) {
        ss << "one= " << collision.one << " two= " << collision.two << " time= " << collision.time;
    }

    void expectEqual(const Collision& expected, const Collision& actual, std::shared_ptr<TestInstance> testInstance) {

        if ((expected.one != actual.one) || (expected.two != actual.two) || (expected.time != actual.time)) {
            std::stringstream ss;
            ss << "expected = ";
            printCollision(expected, ss);
            ss << " actual = ";
            printCollision(actual, ss);
            std::cout << ss.str() << "\n";

            testInstance->assertTrue(false);
        }
    }

    void testHelper(
        const std::vector<Collision>& collisions,
        std::shared_ptr<ReduceVulkanTest> vulkanTest,
        std::shared_ptr<TestInstance> testInstance) {

        auto expected = reduceSerial(collisions);
        auto actualVulkan = vulkanTest->run(collisions);
        auto actualCuda = ReduceCudaTest::run(collisions);

        expectEqual(expected, actualVulkan, testInstance);
        expectEqual(expected, actualCuda, testInstance);
    }
} // end namespace anonymous

ReduceTest::ReduceTest(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool)
    : m_vulkanTest(std::make_shared<ReduceVulkanTest>(physicalDevice, logicalDevice, queue, commandPool, kMaxNumberOfElements)) {}

ReduceTest::~ReduceTest() {}

void ReduceTest::run(std::shared_ptr<TestRunner> testRunner) {

    std::cout << "\n" << TextColour::BLUE << "ReduceTest started" << TextColour::END << "\n";

    std::vector<std::pair<std::string, std::vector<Collision>(*)(uint32_t)>> nameAndFns = {
        {"Basic", generateCollisions}
    };

    for (uint32_t size : kSizes) {
        for (const auto& nameAndFn : nameAndFns) {
            std::ostringstream testName;
            testName << "testReduce_" << nameAndFn.first << "_" << size;
            auto fn = nameAndFn.second;
            testRunner->test(testName.str(), [this, fn, size](auto testInstance) {
                testHelper(fn(size), m_vulkanTest, testInstance);
            });
        }
    }

    std::cout << "\n" << TextColour::PURPLE << "ReduceTest finished" << TextColour::END << "\n";
}
