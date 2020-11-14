#ifndef LOGICAL_DEVICE_H
#define LOGICAL_DEVICE_H

#include <vulkan/vulkan.h>

#include <set>

namespace LogicalDevice {

    VkDevice createLogicalDevice(
        VkPhysicalDevice physicalDevice,
        VkSurfaceKHR surface,
        const std::set<uint32_t>& uniqueQueueFamilies);
}

#endif
