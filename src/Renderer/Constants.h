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

    const std::string kAsteroidModelPath = "models/asteroid.obj";
    const std::string kAsteroidTexturePath = "textures/asteroid_texture.png";

    const std::string kVikingRoomModelPath = "models/viking_room.obj";
    const std::string kVikingRoomTexturePath = "textures/viking_room.png";

    const std::string kMarsModelPath = "models/mars.obj";
    const std::string kMarsTexturePath = "textures/mars_texture.jpg";

    const std::string kMoonModelPath = "models/moon.obj";
    const std::string kMoonTexturePath = "textures/moon_texture.png";

    const std::string kDragonModelPath = "models/dragon.obj";
    const std::string kDragonTexturePath = "textures/dragon_texture.png";

    const std::string kPlasmaModelPath = "models/plasma.obj";
    const std::string kPlasmaTexturePath = "textures/plasma_texture.png";

    const std::vector<const char*> kValidationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    const std::vector<const char*> kDeviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    constexpr bool kEnableValidationLayers = true;
}

#endif
