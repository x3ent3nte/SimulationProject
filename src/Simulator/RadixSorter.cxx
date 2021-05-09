#include <Simulator/RadixSorter.h>

namespace {
    constexpr uint32_t kRadix = 2;
    constexpr uint32_t kMaxLoops = (sizeof(uint32_t) * 8) / kRadix;
} // namespace anonymous

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

bool RadixSorter::needsSorting() {
    return true;
}

void RadixSorter::mapRadixToUVec4(uint32_t radix) {

}

void RadixSorter::scatter(uint32_t radix) {

}

void RadixSorter::sortAtRadix(uint32_t radix) {
    mapRadixToUVec4(radix);
    m_scanner->run(m_currentNumberOfElements);
    scatter(radix);
}

void RadixSorter::sort() {
    bool needsCopyAfterwards = false;

    for (uint32_t i = 0; i < kRadix; ++i) {
        if (needsSorting()) {
            sortAtRadix(i);
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
