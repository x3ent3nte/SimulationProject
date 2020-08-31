#ifndef INSTANCE_H
#define INSTANCE_H

#include <vulkan/vulkan.h>

namespace Instance {

    VkInstance createInstance();

    void setupDebugMessenger(VkInstance instance, VkDebugUtilsMessengerEXT& debugMessenger);

    void DestroyDebugUtilsMessengerEXT(
        VkInstance instance,
        VkDebugUtilsMessengerEXT debugMessenger,
        const VkAllocationCallbacks* pAllocator);
}

#endif
