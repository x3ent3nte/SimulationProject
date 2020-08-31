#ifndef VERTEX_H
#define VERTEX_H

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <array>

struct Vertex {
    glm::vec3 pos;
    glm::vec3 colour;
    glm::vec2 texCoord;

    bool operator==(const Vertex& other) const;

    static VkVertexInputBindingDescription getBindingDescription();

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions();
};

namespace std {
    template<>
    struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const;
    };
}

#endif