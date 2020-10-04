#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <vulkan/vulkan.h>

#include <vector>
#include <string>

namespace Constants {

    constexpr uint32_t kWidth = 800;
    constexpr uint32_t kHeight = 600;

    //const std::string kModelPath = "models/viking_room.obj";
    //const std::string kTexturePath = "textures/viking_room.png";

    const std::string kModelPath = "models/arwing.obj";
    const std::string kTexturePath = "textures/arwing_texture.png";

    constexpr int kMaxFramesInFlight = 2;

    const std::vector<const char*> kValidationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    const std::vector<const char*> kDeviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    const bool kEnableValidationLayers = true;
}

#endif
