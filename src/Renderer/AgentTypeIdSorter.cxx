#include <Renderer/AgentTypeIdSorter.h>

namespace {

} // namespace anonymous

AgentTypeIdSorter::AgentTypeIdSorter(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t maxNumberOfElements,
    size_t descriptorPoolSize) {

    m_physicalDevice = physicalDevice;
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

AgentTypeIdSorterFunction::AgentTypeIdSorterFunction(
    std::shared_ptr<AgentTypeIdSorter> agentTypeIdSorter,
        VkBuffer agentsIn,
        VkBuffer agentsOut,
        uint32_t maxNumberOfAgents) {

    m_agentTypeIdSorter = agentTypeIdSorter;
}

AgentTypeIdSorterFunction::~AgentTypeIdSorterFunction() {

}

std::vector<AgentTypeIdSorter::TypeIdIndex> AgentTypeIdSorterFunction::run(uint32_t numberOfElements) {
    m_agentTypeIdSorter->m_radixSorter->run(numberOfElements);
    return {
        {0, 0},
        {1, numberOfElements / 2}
    };
}
