#ifndef SWAP_CHAIN_H
#define SWAP_CHAIN_H

#include <Renderer/PhysicalDevice.h>

#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace SwapChain {

    void createSwapChain(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkSurfaceKHR surface,
        GLFWwindow* window,
        VkFormat& swapChainImageFormat,
        VkExtent2D& swapChainExtent,
        VkSwapchainKHR& swapChain,
        std::vector<VkImage>& swapChainImages,
        std::vector<VkImageView>& swapChainImageViews);

}

#endif
