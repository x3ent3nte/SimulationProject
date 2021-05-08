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

/*
bool needsSorting() {
    // TODO
    return true;
}

glm::vec4 extractOffsets() {
    // TODO
    glm::vec4 extracted = {0,0,0,0};

    return {0,};
}

void sortAtRadix(uint32_t radix, uint32_t numberOfElements) {
    mapRadixToVec4(radix, m_scanner->m_dataBuffer, numberOfElements);
    m_scanner->run(numberOfElements);
    glm::vec4 offsets = extractOffsets();
}

void sort() {
    for (uint32_t i = 0; i < maxLoops; ++i) {
        if (needsSorting()) {
            sortAtRadix(i);
        } else {
            return;
        }
    }
}
*/
std::vector<AgentTypeIdSorter::TypeIdIndex> AgentTypeIdSorter::run(VkBuffer agents, uint32_t numberOfElements) {

    //mapAgentRenderInfoToTypeInfoAndIndex();

    //sort();

    //scatterAgentRenderInfo();

    return {
        {0, 0},
        {1, numberOfElements / 2}
    };
}
