#include <Simulator/AgentSorter.h>

#include <Simulator/MapAgentToXUtil.h>
#include <Simulator/MapXToAgentUtil.h>

AgentSorter::AgentSorter(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t numberOfElements)
    : m_insertionSorter(std::make_shared<InsertionSorter>(
        physicalDevice,
        logicalDevice,
        queue,
        commandPool,
        numberOfElements)) {

}

AgentSorter::~AgentSorter() {

}

void AgentSorter::run(float timeDelta, uint32_t numberOfElements) {

}
