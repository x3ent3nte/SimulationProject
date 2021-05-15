#ifndef MESH_H
#define MESH_H

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

struct SubMeshInfo {
    int32_t vertexOffset;
    uint32_t indexOffset;
    uint32_t numberOfIndexes;
    float radius;
};

class Mesh {

public:

    std::vector<SubMeshInfo> m_subMeshInfos;

    Mesh(
        const std::vector<std::string>& modelPaths,
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool);

    virtual ~Mesh();

private:

    VkDevice m_logicalDevice;

    VkBuffer m_vertexesBuffer;
    VkBuffer m_indicesBuffer;

    VkDeviceMemory m_vertexesDeviceMemory;
    VkDeviceMemory m_indicesDeviceMemory;
};

#endif
