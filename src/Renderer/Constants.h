#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <vulkan/vulkan.h>

#include <vector>
#include <string>

namespace Constants {

    const std::string kFreyjaModelPath = "models/freyja.obj";
    const std::string kFreyjaTexturePath = "textures/freyja_texture.png";

    const std::string kArwingModelPath = "models/arwing.obj";
    const std::string kArwingTexturePath = "textures/arwing_texture.png";

    const std::string kVikingRoomModelPath = "models/viking_room.obj";
    const std::string kVikingRoomTexturePath = "textures/viking_room.png";

    const std::vector<const char*> kValidationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    const std::vector<const char*> kDeviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    constexpr bool kEnableValidationLayers = true;
}

#endif
