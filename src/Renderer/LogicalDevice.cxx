#include <Renderer/LogicalDevice.h>

#include <Renderer/Constants.h>

#include <vector>
#include <stdexcept>

VkDevice LogicalDevice::createLogicalDevice(
    VkPhysicalDevice physicalDevice,
    const std::set<uint32_t>& uniqueQueueFamilies) {

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

    float queuePriority = 1.0f;

    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;
    deviceFeatures.sampleRateShading = VK_TRUE;
    deviceFeatures.multiDrawIndirect = VK_TRUE;
    deviceFeatures.drawIndirectFirstInstance = VK_TRUE;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = 0;

    createInfo.enabledExtensionCount = static_cast<uint32_t>(Constants::kDeviceExtensions.size());
    createInfo.ppEnabledExtensionNames = Constants::kDeviceExtensions.data();

    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    if (Constants::kEnableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(Constants::kValidationLayers.size());
        createInfo.ppEnabledLayerNames = Constants::kValidationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    VkDevice logicalDevice;
    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &logicalDevice) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device");
    }
    return logicalDevice;
}
