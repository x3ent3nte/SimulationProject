#ifndef PHYSICAL_DEVICE_H
#define PHYSICAL_DEVICE_H

#include <vulkan/vulkan.h>

#include <vector>

namespace PhysicalDevice {

    struct QueueFamilyIndices {
        uint32_t m_graphicsFamily;
        bool m_hasGraphicsFamily;
        uint32_t m_presentFamily;
        bool m_hasPresentFamily;

        bool isComplete();
    };

    struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR m_capabilities;
        std::vector<VkSurfaceFormatKHR> m_formats;
        std::vector<VkPresentModeKHR> m_presentModes;
    };

    VkPhysicalDevice pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface);

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface);

    size_t findComputeQueueIndex(VkPhysicalDevice device);

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);

    uint32_t findMemoryType(
        VkPhysicalDevice physicalDevice,
        uint32_t typeFilter,
        VkMemoryPropertyFlags properties);

    VkSampleCountFlagBits getMaxUsableSampleCount(VkPhysicalDevice device);
}

#endif
