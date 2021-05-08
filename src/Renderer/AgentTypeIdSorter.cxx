#include <Renderer/AgentTypeIdSorter.h>

namespace {

constexpr uint32_t radix = 2;
constexpr uint32_t maxLoops = (sizeof(uint32_t) * 8) / radix;

} // namespace anonymous

AgentTypeIdSorter::AgentTypeIdSorter(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t maxNumberOfElements) {

    m_logicalDevice = logicalDevice;
    m_queue = queue;
    m_commandPool = commandPool;

    m_scanner = std::make_shared<Scanner<glm::uvec4>>(
        physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        maxNumberOfElements);
}

AgentTypeIdSorter::~AgentTypeIdSorter() {

}


bool AgentTypeIdSorter::needsSorting() {
    // TODO
    return true;
}

glm::vec4 AgentTypeIdSorter::extractOffsets() {
    // TODO
    glm::vec4 endValue = {0,0,0,0};

    const uint32_t y = endValue.x;
    const uint32_t z = y + endValue.y;
    const uint32_t w = z + endValue.z;
    return {0, y, z, w};
}

void AgentTypeIdSorter::mapRadixToVec4(uint32_t radix, VkBuffer data, uint32_t numberOfElements) {

}

void AgentTypeIdSorter::scatterInfo(uint32_t radix, const glm::uvec4& offsets) {

}

void AgentTypeIdSorter::sortAtRadix(uint32_t radix, uint32_t numberOfElements) {
    mapRadixToVec4(radix, m_scanner->m_dataBuffer, numberOfElements);
    m_scanner->run(numberOfElements);
    glm::vec4 offsets = extractOffsets();
    scatterInfo(radix, offsets);
}

void AgentTypeIdSorter::sort(uint32_t numberOfElements) {
    for (uint32_t i = 0; i < maxLoops; ++i) {
        if (needsSorting()) {
            sortAtRadix(i, numberOfElements);
        } else {
            return;
        }
    }
}

void AgentTypeIdSorter::mapAgentRenderInfoToTypeInfoAndIndex() {

}

void AgentTypeIdSorter::scatterTypeInfoAndIndexToAgentRenderInfo() {

}

std::vector<AgentTypeIdSorter::TypeIdIndex> AgentTypeIdSorter::calculateTypeIdIndexes(uint32_t numberOfElements) {
    return {
        {0, 0},
        {1, numberOfElements / 2}
    };
}

std::vector<AgentTypeIdSorter::TypeIdIndex> AgentTypeIdSorter::run(VkBuffer agents, uint32_t numberOfElements) {

    mapAgentRenderInfoToTypeInfoAndIndex();

    sort(numberOfElements);

    scatterTypeInfoAndIndexToAgentRenderInfo();

    return calculateTypeIdIndexes(numberOfElements);
}
