#include <Renderer/Model.h>

#include <Renderer/Vertex.h>
#include <Renderer/Image.h>
#include <Utils/Buffer.h>
#include <Utils/Utils.h>

#include <iostream>

Model::Model(
    const std::string& objectName,
    const std::string& textureName,
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkCommandPool commandPool,
    VkQueue queue) {

    m_logicalDevice = logicalDevice;
    m_radius = 0.0f;

    std::vector<Vertex> vertexes;
    std::vector<uint32_t> indices;

    Utils::loadModel(vertexes, indices, objectName);

    for (const Vertex& vertex : vertexes) {
        float mag = glm::length(vertex.pos);
        if (mag > m_radius) {
            m_radius = mag;
        }
    }

    std::cout << "Model " << objectName << " radius = " << m_radius << "\n";

    m_numberOfIndices = indices.size();

    Buffer::createBufferWithData(
        vertexes.data(),
        sizeof(Vertex) * vertexes.size(),
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

size_t Model::numberOfIndices() const {
    return m_numberOfIndices;
}

float Model::radius() const {
    return m_radius;
}
