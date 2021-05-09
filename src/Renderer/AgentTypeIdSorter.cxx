#include <Renderer/AgentTypeIdSorter.h>

namespace {

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

    m_radixSorter = std::make_shared<RadixSorter>(
        physicalDevice,
        m_logicalDevice,
        m_queue,
        m_commandPool,
        maxNumberOfElements);
}

AgentTypeIdSorter::~AgentTypeIdSorter() {

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

    m_radixSorter->run(numberOfElements);

    scatterTypeInfoAndIndexToAgentRenderInfo();

    return calculateTypeIdIndexes(numberOfElements);
}
