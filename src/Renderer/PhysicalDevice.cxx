#include <Renderer/PhysicalDevice.h>

#include <Renderer/Constants.h>

#include <iostream>
#include <vector>
#include <set>

namespace {

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(Constants::kDeviceExtensions.begin(), Constants::kDeviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    int rateDeviceSuitability(VkPhysicalDevice device, VkSurfaceKHR surface) {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);

        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        std::cout << "Device Name: " << deviceProperties.deviceName << "\n";

        int score = 0;

        auto indices = PhysicalDevice::findQueueFamilies(device, surface);

        bool extensionSupport = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionSupport) {
            PhysicalDevice::SwapChainSupportDetails swapChainSupport = PhysicalDevice::querySwapChainSupport(device, surface);
            swapChainAdequate = (!swapChainSupport.m_formats.empty()) && (!swapChainSupport.m_presentModes.empty());
        }

        if ((!indices.isComplete()) || (!deviceFeatures.geometryShader)
            || (!extensionSupport) || (!swapChainAdequate)
            || (!deviceFeatures.samplerAnisotropy)) {

            return score;
        }

        if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score = 1000;
        }

        score += deviceProperties.limits.maxImageDimension2D;

        return score;
    }

} // namespace anonymous

bool PhysicalDevice::QueueFamilyIndices::isComplete() {
    return m_hasGraphicsFamily && m_hasPresentFamily && m_hasComputeFamily;
}

VkPhysicalDevice PhysicalDevice::pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface) {

    uint32_t deviceCount;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    std::cout << "Found " << deviceCount << " devices\n";

    if (deviceCount == 0) {
        throw std::runtime_error("Failed to find GPUs with Vulkan support");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    VkPhysicalDevice physicalDevice;

    int highestScore = 0;
    for (const auto& device : devices) {
        std::cout << "Device: " << device << "\n";

        int score = rateDeviceSuitability(device, surface);
        std::cout << "Score: " << score << "\n";
        if (score > highestScore) {
            highestScore = score;
            physicalDevice = device;
            //msaaSamples = getMaxUsableSampleCount(physicalDevice);
            break;
        }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("Failed to find a suitable GPU");
    }

    return physicalDevice;
}

PhysicalDevice::QueueFamilyIndices PhysicalDevice::findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {
    QueueFamilyIndices indices{0, false, 0, false, 0, false};

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.m_graphicsFamily = i;
            indices.m_hasGraphicsFamily = true;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

        if (presentSupport) {
            indices.m_presentFamily = i;
            indices.m_hasPresentFamily = true;
        }

        if (indices.m_hasPresentFamily && indices.m_hasGraphicsFamily
            && (indices.m_graphicsFamily == indices.m_presentFamily)) {

            break;
        }

        i += 1;
    }

    i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if ((queueFamily.queueCount > 0) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            indices.m_computeFamily = i;
            indices.m_hasComputeFamily = true;
        }
        i += 1;
    }

    return indices;
}

size_t PhysicalDevice::findComputeQueueIndex(VkPhysicalDevice device) {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    size_t i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if ((queueFamily.queueCount > 0) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            return i;
        }
        i += 1;
    }

    return 0;
}

PhysicalDevice::SwapChainSupportDetails PhysicalDevice::querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface) {
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.m_capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

    if (formatCount != 0) {
        details.m_formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.m_formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

    if (presentModeCount != 0) {
        details.m_presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.m_presentModes.data());
    }

    return details;
}

uint32_t PhysicalDevice::findMemoryType(
    VkPhysicalDevice physicalDevice,
    uint32_t typeFilter,
    VkMemoryPropertyFlags properties) {

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type");
}

VkSampleCountFlagBits PhysicalDevice::getMaxUsableSampleCount(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(device, &physicalDeviceProperties);

    VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts
        & physicalDeviceProperties.limits.framebufferDepthSampleCounts;

    if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
    if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
    if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
    if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
    if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
    if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

    return VK_SAMPLE_COUNT_1_BIT;
}
