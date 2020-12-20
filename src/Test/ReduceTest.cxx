#include <Test/ReduceTest.h>

#include <Test/TestUtils.h>
#include <Simulator/Agent.h>
#include <Utils/Timer.h>

#include <vector>
#include <iostream>
#include <sstream>

namespace {
    constexpr uint32_t kMaxNumberOfElements = 512 * 1024;

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
            collisions[i] = {i, i + 1, float(kMaxNumberOfElements - i)};
        }

        collisions[kMaxNumberOfElements / 2] = {123, 321, 0};
        return collisions;
    }

    void printCollision(const Collision& collision, std::stringstream& ss) {
        ss << "one= " << collision.one << " two= " << collision.two << " time= " << collision.time;
    }

    void expectEqual(const Collision& expected, const Collision& actual) {

        std::stringstream ss;
            ss << "expected = ";
            printCollision(expected, ss);
            ss << " actual= ";
            printCollision(actual, ss);
            std::cout << ss.str() << "\n";

        if ((expected.one != actual.one) || (expected.two != actual.two) || (expected.time != actual.time)) {
            TestUtils::assertTrue(false);
        }
    }

    void testHelper(
        const std::vector<Collision>& collisions,
        std::shared_ptr<ReduceVulkanTest> vulkanTest) {

        auto expected = reduceSerial(collisions);
        auto actualVulkan = vulkanTest->run(collisions);
        auto actualCuda = ReduceCudaTest::run(collisions);

        expectEqual(expected, actualVulkan);
        expectEqual(expected, actualCuda);
    }

    void testBasic(std::shared_ptr<ReduceVulkanTest> vulkanTest) {
        testHelper(generateCollisions(kMaxNumberOfElements), vulkanTest);
    }
} // end namespace anonymous

ReduceTest::ReduceTest(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool)
    : m_vulkanTest(std::make_shared<ReduceVulkanTest>(physicalDevice, logicalDevice, queue, commandPool, kMaxNumberOfElements)) {}

ReduceTest::~ReduceTest() {}

void ReduceTest::run() {
    TestUtils::testRunner("testReduceBasic", [this]() { testBasic(m_vulkanTest); });
}
