#ifndef AGENT_TYPEID_SORTER_H
#define AGENT_TYPEID_SORTER_H

#include <Simulator/Scanner.h>

#include <vulkan/vulkan.h>

#include <memory>
#include <vector>

class AgentTypeIdSorter {

private:

    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    std::shared_ptr<Scanner<glm::vec4>> m_scanner;

public:

    AgentTypeIdSorter(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        uint32_t maxNumberOfElements);

    virtual ~AgentTypeIdSorter();

    struct TypeIdIndex {
        uint32_t typeId;
        uint32_t index;
    };

    std::vector<TypeIdIndex> run(VkBuffer agents, uint32_t numberOfElements);
};

#endif
