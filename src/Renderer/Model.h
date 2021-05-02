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

private:

    VkDevice m_logicalDevice;

    VkBuffer m_vertexesBuffer;
    VkDeviceMemory m_vertexesDeviceMemory;

    VkBuffer m_indicesBuffer;
    VkDeviceMemory m_indicesDeviceMemory;
};

#endif
