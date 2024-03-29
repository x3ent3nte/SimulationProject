#include <Renderer/Vertex.h>

#include <Simulator/Agent.h>

bool Vertex::operator==(const Vertex& other) const {
    return (pos == other.pos) && (normal == other.normal) && (colour == other.colour) && (texCoord == other.texCoord);
}

std::array<VkVertexInputBindingDescription, 1> Vertex::getBindingDescriptions() {
    std::array<VkVertexInputBindingDescription, 1> bindingDescriptions;

    bindingDescriptions[0].binding = 0;
    bindingDescriptions[0].stride = sizeof(Vertex);
    bindingDescriptions[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    return bindingDescriptions;
}

std::array<VkVertexInputAttributeDescription, 4> Vertex::getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, normal);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Vertex, colour);

    attributeDescriptions[3].binding = 0;
    attributeDescriptions[3].location = 3;
    attributeDescriptions[3].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[3].offset = offsetof(Vertex, texCoord);

    return attributeDescriptions;
}

size_t std::hash<Vertex>::operator()(Vertex const& vertex) const {
    return ((hash<glm::vec3>()(vertex.pos) ^
        (hash<glm::vec3>()(vertex.colour) << 1)) >> 1) ^
        (hash<glm::vec2>()(vertex.texCoord) << 1);
}
