#include <Renderer/AgentTypeIdSorter.h>

std::vector<AgentTypeIdSorter::TypeIdIndex> AgentTypeIdSorter::run(VkBuffer agents, uint32_t numberOfAgents) {
    return {
        {0, 0},
        {1, numberOfAgents / 2}
    };
}
