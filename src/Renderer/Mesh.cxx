#include <Renderer/Mesh.h>

#include <Renderer/Vertex.h>
#include <Utils/Buffer.h>
#include <Utils/Utils.h>

#include <iostream>

namespace {

std::vector<SubMeshInfo> initializeMesh(
    const std::vector<std::string>& modelPaths,
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    VkBuffer vertexesBuffer,
    VkDeviceMemory vertexesDeviceMemory,
    VkBuffer indicesBuffer,
    VkDeviceMemory indicesDeviceMemory) {

    const size_t numberOfSubMeshes = modelPaths.size();

    std::vector<SubMeshInfo> subMeshInfos(numberOfSubMeshes);
    std::vector<Vertex> vertexes;
    std::vector<uint32_t> indices;

    int32_t vertexOffset = 0;
    uint32_t indexOffset = 0;
    for (int i = 0; i < numberOfSubMeshes; ++i) {
        Utils::loadModel(vertexes, indices, modelPaths[i]);

        float radius = 0.0f;
        for (const Vertex& vertex : vertexes) {
            float mag = glm::length(vertex.pos);
            if (mag > radius) {
                radius = mag;
            }
        }

        subMeshInfos[i] = {
            vertexOffset,
            indexOffset,
            static_cast<uint32_t>(indices.size()) - indexOffset,
            radius
        };

        vertexOffset = vertexes.size();
        indexOffset = indices.size();
    }

    std::cout << "Vertexes Size " << vertexes.size() << "\n";
    std::cout << "Indices Size " << indices.size() << "\n";

    Buffer::createBufferWithData(
        vertexes.data(),
        sizeof(Vertex) * vertexes.size(),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        physicalDevice,
        logicalDevice,
        commandPool,
        queue,
        vertexesBuffer,
        vertexesDeviceMemory);

    Buffer::createBufferWithData(
        indices.data(),
        sizeof(uint32_t) * indices.size(),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        physicalDevice,
        logicalDevice,
        commandPool,
        queue,
        indicesBuffer,
        indicesDeviceMemory);

    return subMeshInfos;
}

} // namespace anonymous

Mesh::Mesh(
    const std::vector<std::string>& modelPaths,
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool) {

    m_logicalDevice = logicalDevice;
    m_subMeshInfos = initializeMesh(
        modelPaths,
        physicalDevice,
        m_logicalDevice,
        queue,
        commandPool,
        m_vertexesBuffer,
        m_vertexesDeviceMemory,
        m_indicesBuffer,
        m_indicesDeviceMemory);
}

Mesh::~Mesh() {

    std::cout << "Mesh Destructor Called\n";

    vkFreeMemory(m_logicalDevice, m_indicesDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_indicesBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_vertexesDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_vertexesBuffer, nullptr);
}
