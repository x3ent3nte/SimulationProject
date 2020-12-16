#ifndef AGENT_SORTER_H
#define AGENT_SORTER_H

#include <Simulator/InsertionSorter.h>

#include <vulkan/vulkan.h>

#include <memory>

class AgentSorter {

private:

    std::shared_ptr<InsertionSorter> m_insertionSorter;

public:

    AgentSorter(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        uint32_t numberOfElements);

    virtual ~AgentSorter();

    void run(uint32_t numberOfElements);

};

#endif
