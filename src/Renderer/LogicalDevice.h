#ifndef LOGICAL_DEVICE_H
#define LOGICAL_DEVICE_H

#include <vulkan/vulkan.h>

namespace LogicalDevice {
    void createLogicalDevice(
        VkPhysicalDevice physicalDevice,
        VkSurfaceKHR surface,
        VkDevice& logicalDevice,
        VkQueue& graphicsQueue,
        VkQueue& presentQueue,
        VkQueue& computeQueue);
}

#endif
