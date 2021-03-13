#include <Test/ScanTest.h>

#include <Test/ScanCudaTest.cuh>
#include <Test/TestUtils.h>
#include <Utils/Timer.h>

namespace {

    constexpr uint32_t kMaxNumberOfElements = 64 * 1024;

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
        std::shared_ptr<ScanVulkanTest> vulkanTest) {
        auto expected = serialScan(data);

        auto actualVulkan = vulkanTest->run(data);
        //TestUtils::assertEqual(expected, actualVulkan);

        auto actualCuda = ScanCudaTest::run(data);
        TestUtils::assertEqual(expected, actualCuda);
    }

    void testAllOnes(std::shared_ptr<ScanVulkanTest> vulkanTest) {
        std::vector<int> data(1024 * 16);
        for (int i = 0; i < data.size(); ++i) {
            data[i] = 1;
        }

        testHelper(data, vulkanTest);
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
    TestUtils::testRunner("testScanBasic", [this]() { testAllOnes(m_vulkanTest); });
}
