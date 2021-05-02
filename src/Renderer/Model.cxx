#include <Renderer/Model.h>

#include <Utils/Buffer.h>
#include <Utils/Utils.h>
#include <Renderer/Vertex.h>

Model::Model(
    const std::string& objectName,
    const std::string& textureName,
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkCommandPool commandPool,
    VkQueue queue) {

    m_logicalDevice = logicalDevice;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    Utils::loadModel(vertices, indices, objectName);

    Buffer::createBufferWithData(
        vertices.data(),
        sizeof(Vertex) * vertices.size(),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        physicalDevice,
        m_logicalDevice,
        commandPool,
        queue,
        m_vertexesBuffer,
        m_vertexesDeviceMemory);

    Buffer::createBufferWithData(
        indices.data(),
        sizeof(uint32_t) * indices.size(),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        physicalDevice,
        m_logicalDevice,
        commandPool,
        queue,
        m_indicesBuffer,
        m_indicesDeviceMemory);
}

Model::~Model() {
    vkDestroyBuffer(m_logicalDevice, m_indicesBuffer, nullptr);
    vkFreeMemory(m_logicalDevice, m_indicesDeviceMemory, nullptr);

    vkDestroyBuffer(m_logicalDevice, m_vertexesBuffer, nullptr);
    vkFreeMemory(m_logicalDevice, m_vertexesDeviceMemory, nullptr);
}
