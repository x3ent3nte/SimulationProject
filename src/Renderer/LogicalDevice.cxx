#include <Renderer/LogicalDevice.h>

#include <Renderer/PhysicalDevice.h>
#include <Renderer/Constants.h>

#include <vector>
#include <set>
#include <stdexcept>

void LogicalDevice::createLogicalDevice(
        VkPhysicalDevice physicalDevice,
        VkSurfaceKHR surface,
        VkDevice& logicalDevice,
        VkQueue& graphicsQueue,
        VkQueue& presentQueue) {

        PhysicalDevice::QueueFamilyIndices indices = PhysicalDevice::findQueueFamilies(physicalDevice, surface);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.m_graphicsFamily, indices.m_presentFamily, 2};

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

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &logicalDevice) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create logical device");
        }

        vkGetDeviceQueue(logicalDevice, indices.m_graphicsFamily, 0, &graphicsQueue);
        vkGetDeviceQueue(logicalDevice, indices.m_presentFamily, 0, &presentQueue);
    }
