#ifndef MODEL_H
#define MODEL_H

#include <vulkan/vulkan.h>

#include <string>

class Model {

public:

    Model(
        const std::string& objectName,
        const std::string& textureName,
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkQueue queue);

    virtual ~Model();

    size_t numberOfIndices() const;
    float radius() const;

    VkBuffer m_vertexesBuffer;
    VkBuffer m_indicesBuffer;

private:

    size_t m_numberOfIndices;
    float m_radius;

    VkDevice m_logicalDevice;
    VkDeviceMemory m_vertexesDeviceMemory;
    VkDeviceMemory m_indicesDeviceMemory;
};

#endif
