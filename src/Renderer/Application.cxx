#include <Renderer/Application.h>

#include <Renderer/Instance.h>
#include <Renderer/Surface.h>
#include <Renderer/PhysicalDevice.h>
#include <Renderer/LogicalDevice.h>
#include <Renderer/KeyboardControl.h>
#include <Renderer/Constants.h>

#include <set>
#include <iostream>
#include <stdexcept>

Application::Application() {

    m_keyboardControl = std::make_shared<KeyboardControl>();
    m_window = Surface::createWindow(m_keyboardControl);

    m_instance = Instance::createInstance();
    Instance::setupDebugMessenger(m_instance, m_debugMessenger);

    m_surface = Surface::createSurface(m_instance, m_window->m_window);

    m_physicalDevice = PhysicalDevice::pickPhysicalDevice(m_instance, m_surface);
    PhysicalDevice::QueueFamilyIndices indices = PhysicalDevice::findQueueFamilies(m_physicalDevice, m_surface);

    uint32_t computeQueueIndex = 2;
    std::set<uint32_t> uniqueQueueFamilies = {indices.m_graphicsFamily, indices.m_presentFamily, computeQueueIndex};
    m_logicalDevice = LogicalDevice::createLogicalDevice(m_physicalDevice, m_surface, uniqueQueueFamilies);

    vkGetDeviceQueue(m_logicalDevice, indices.m_graphicsFamily, 0, &m_graphicsQueue);
    vkGetDeviceQueue(m_logicalDevice, indices.m_presentFamily, 0, &m_presentQueue);
    vkGetDeviceQueue(m_logicalDevice, computeQueueIndex, 0, &m_computeQueue);
}

Application::~Application() {

    if(Constants::kEnableValidationLayers) {
        Instance::DestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
    }

    vkDestroyDevice(m_logicalDevice, nullptr);

    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);

    vkDestroyInstance(m_instance, nullptr);

    glfwDestroyWindow(m_window->m_window);
    glfwTerminate();
}

int Application::run() {

    std::shared_ptr<Renderer> renderer = Renderer::create(
        m_instance,
        m_window,
        m_surface,
        m_keyboardControl,
        m_physicalDevice,
        m_logicalDevice,
        m_graphicsQueue,
        m_presentQueue);

    m_prevTime = std::chrono::high_resolution_clock::now();

    try {
        while (!glfwWindowShouldClose(m_window->m_window)) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - m_prevTime).count();

            glfwPollEvents();
            renderer->render(time);

            m_prevTime = currentTime;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }

    vkDeviceWaitIdle(m_logicalDevice);

    return EXIT_SUCCESS;
}
