#include <Simulator/RadixSorter.h>

namespace {
    constexpr uint32_t kRadix = 2;
    constexpr uint32_t kNumberOfBits = sizeof(uint32_t) * 8;
} // namespace anonymous

namespace RadixSorterUtil {
    constexpr size_t kXDim = 512;
}

RadixSorter::RadixSorter(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t maxNumberOfElements) {

    m_logicalDevice = logicalDevice;
    m_queue = queue;
    m_commandPool = commandPool;

    m_currentNumberOfElements = maxNumberOfElements;
    createCommandBuffers();

    m_scanner = std::make_shared<Scanner<glm::uvec4>>(
        physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        maxNumberOfElements);
}

RadixSorter::~RadixSorter() {
    destroyCommandBuffers();
}

void RadixSorter::destroyCommandBuffers() {

}

void RadixSorter::createCommandBuffers() {

}

void RadixSorter::createCommandBuffersIfNecessary(uint32_t numberOfElements) {
    if (numberOfElements != m_currentNumberOfElements) {
        destroyCommandBuffers();
        m_currentNumberOfElements = numberOfElements;
        createCommandBuffers();
    }
}

void RadixSorter::copyBuffers() {

}

void RadixSorter::setRadix(uint32_t radix) {

}

bool RadixSorter::needsSorting() {
    return true;
}

void RadixSorter::mapRadixToUVec4() {

}

void RadixSorter::scatter() {

}

void RadixSorter::sortAtRadix(uint32_t radix) {
    setRadix(radix);
    mapRadixToUVec4();
    m_scanner->run(m_currentNumberOfElements);
    scatter();
}

void RadixSorter::sort() {
    bool needsCopyAfterwards = false;

    for (uint32_t radix = 0; radix < kNumberOfBits; radix += kRadix) {
        if (needsSorting()) {
            sortAtRadix(radix);
            needsCopyAfterwards = !needsCopyAfterwards;
        } else {
            break;
        }
    }

    if (needsCopyAfterwards) {
        copyBuffers();
    }
}

void RadixSorter::run(uint32_t numberOfElements) {
    createCommandBuffersIfNecessary(numberOfElements);
    sort();
}
