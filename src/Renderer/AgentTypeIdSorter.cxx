#include <Renderer/AgentTypeIdSorter.h>

namespace {

const uint32_t maxLoops = (sizeof(int32_t) * 8) / 2;

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

    m_scanner = std::make_shared<Scanner<glm::vec4>>(
        physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        maxNumberOfElements);
}

AgentTypeIdSorter::~AgentTypeIdSorter() {

}

/*
void sort() {
    for (uint32_t i = 0; i < maxLoops; ++i) {
        sortAtRadix(i);
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
