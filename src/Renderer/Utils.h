#ifndef UTILS_H
#define UTILS_H

#include <Renderer/Vertex.h>

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

namespace Utils {

    void loadModel(
        std::vector<Vertex>& vertices,
        std::vector<uint32_t>& indices,
        const std::string& modelPath);

    std::vector<char> readFile(const std::string& filename);

    VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code);
}

#endif
