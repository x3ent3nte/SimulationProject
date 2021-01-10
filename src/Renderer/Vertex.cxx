#include <Renderer/Vertex.h>

#include <Simulator/Agent.h>

bool Vertex::operator==(const Vertex& other) const {
    return (pos == other.pos) && (normal == other.normal) && (colour == other.colour) && (texCoord == other.texCoord);
}

std::array<VkVertexInputBindingDescription, 2> Vertex::getBindingDescriptions() {
    std::array<VkVertexInputBindingDescription, 2> bindingDescriptions;

    bindingDescriptions[0].binding = 0;
    bindingDescriptions[0].stride = sizeof(Vertex);
    bindingDescriptions[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    bindingDescriptions[1].binding = 1;
    bindingDescriptions[1].stride = sizeof(AgentPositionAndRotation);
    bindingDescriptions[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    return bindingDescriptions;
}

std::array<VkVertexInputAttributeDescription, 6> Vertex::getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 6> attributeDescriptions{};

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

    attributeDescriptions[4].binding = 1;
    attributeDescriptions[4].location = 4;
    attributeDescriptions[4].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[4].offset = offsetof(AgentPositionAndRotation, position);

    attributeDescriptions[5].binding = 1;
    attributeDescriptions[5].location = 5;
    attributeDescriptions[5].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[5].offset = offsetof(AgentPositionAndRotation, rotation);

    return attributeDescriptions;
}

size_t std::hash<Vertex>::operator()(Vertex const& vertex) const {
    return ((hash<glm::vec3>()(vertex.pos) ^
        (hash<glm::vec3>()(vertex.colour) << 1)) >> 1) ^
        (hash<glm::vec2>()(vertex.texCoord) << 1);
}
