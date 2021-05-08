#ifndef AGENT_TYPEID_SORTER_H
#define AGENT_TYPEID_SORTER_H

#include <vulkan/vulkan.h>

#include <vector>

class AgentTypeIdSorter {

private:

public:

    struct TypeIdIndex {
        int typeId;
        uint32_t index;
    };

    std::vector<TypeIdIndex> run(VkBuffer agents, uint32_t numberOfAgents);
};

#endif
