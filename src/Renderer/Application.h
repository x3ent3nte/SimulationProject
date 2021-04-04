#ifndef APPLICATION_H
#define APPLICATION_H

#include <Renderer/Renderer.h>
#include <Simulator/Simulator.h>

#include <vulkan/vulkan.h>

#include <memory>
#include <chrono>

class Application {

private:

    std::shared_ptr<KeyboardControl> m_keyboardControl;
    std::shared_ptr<Surface::Window> m_window;

    VkInstance m_instance;
    VkDebugUtilsMessengerEXT m_debugMessenger;

    VkSurfaceKHR m_surface;

    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_logicalDevice;

    VkQueue m_graphicsQueue;
    VkQueue m_presentQueue;
    VkQueue m_computeQueue;

    VkCommandPool m_graphicsCommandPool;
    VkCommandPool m_computeCommandPool;

    std::chrono::time_point<std::chrono::high_resolution_clock> m_prevTime;

public:

    Application();

    virtual ~Application();

    int run();
};

#endif
