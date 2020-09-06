#ifndef VERTEX_H
#define VERTEX_H

#include <vulkan/vulkan.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <array>

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

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
