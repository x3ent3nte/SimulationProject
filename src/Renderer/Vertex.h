#ifndef VERTEX_H
#define VERTEX_H

#include <vulkan/vulkan.h>

#include <Renderer/MyGLM.h>

#include <array>

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec3 cameraPosition;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 colour;
    glm::vec2 texCoord;

    bool operator==(const Vertex& other) const;

    static std::array<VkVertexInputBindingDescription, 2> getBindingDescriptions();

    static std::array<VkVertexInputAttributeDescription, 6> getAttributeDescriptions();
};

namespace std {
    template<>
    struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const;
    };
}

#endif
