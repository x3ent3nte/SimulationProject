#ifndef AGENT_TYPEID_SORTER_H
#define AGENT_TYPEID_SORTER_H

#include <Simulator/Scanner.h>

#include <vulkan/vulkan.h>

#include <memory>
#include <vector>

class AgentTypeIdSorter {

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

private:

    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    std::shared_ptr<Scanner<glm::uvec4>> m_scanner;

    bool needsSorting();
    glm::vec4 extractOffsets();
    void mapRadixToVec4(uint32_t radix, VkBuffer data, uint32_t numberOfElements);
    void scatterInfo(uint32_t radix, const glm::uvec4& offsets);
    void sortAtRadix(uint32_t radix, uint32_t numberOfElements);
    void sort(uint32_t numberOfElements);
    void mapAgentRenderInfoToTypeInfoAndIndex();
    void scatterTypeInfoAndIndexToAgentRenderInfo();
    std::vector<TypeIdIndex> calculateTypeIdIndexes(uint32_t numberOfElements);
};

#endif
