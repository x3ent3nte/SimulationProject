#include <Renderer/Application.h>

#include <Renderer/Instance.h>
#include <Renderer/Surface.h>
#include <Renderer/PhysicalDevice.h>
#include <Renderer/LogicalDevice.h>
#include <Renderer/KeyboardControl.h>
#include <Renderer/Constants.h>
#include <Renderer/Command.h>
#include <Test/TestApplication.h>
#include <Utils/Timer.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

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

    std::set<uint32_t> uniqueQueueFamilies = {indices.m_graphicsFamily, indices.m_presentFamily, indices.m_computeFamily};
    m_logicalDevice = LogicalDevice::createLogicalDevice(m_physicalDevice, uniqueQueueFamilies);

    vkGetDeviceQueue(m_logicalDevice, indices.m_graphicsFamily, 0, &m_graphicsQueue);
    vkGetDeviceQueue(m_logicalDevice, indices.m_presentFamily, 0, &m_presentQueue);
    vkGetDeviceQueue(m_logicalDevice, indices.m_computeFamily, 0, &m_computeQueue);

    m_commandPool = Command::createCommandPool(m_logicalDevice, indices.m_graphicsFamily);
    m_computeCommandPool = Command::createCommandPool(m_logicalDevice, indices.m_computeFamily);
}

Application::~Application() {

    vkDestroyCommandPool(m_logicalDevice, m_commandPool, nullptr);
    vkDestroyCommandPool(m_logicalDevice, m_computeCommandPool, nullptr);

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

    {
        TestApplication(m_physicalDevice, m_logicalDevice, m_computeQueue, m_computeCommandPool).run();
    }

    auto connector = std::make_shared<Connector>(m_physicalDevice, m_logicalDevice, m_commandPool, m_graphicsQueue);
    auto simulator = std::make_shared<Simulator>(m_physicalDevice, m_logicalDevice, m_computeQueue, m_computeCommandPool, connector, Constants::kNumberOfAgents);
    simulator->simulate();

    std::shared_ptr<Renderer> renderer = Renderer::create(
        m_instance,
        m_window,
        m_surface,
        m_keyboardControl,
        m_physicalDevice,
        m_logicalDevice,
        m_graphicsQueue,
        m_presentQueue,
        m_commandPool,
        connector);

    m_prevTime = std::chrono::high_resolution_clock::now();

    int numFramesRendered = 0;
    try {
        while (!glfwWindowShouldClose(m_window->m_window)) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - m_prevTime).count();

            glfwPollEvents();
            //Timer timer("Render Frame " + numFramesRendered);
            renderer->render(time);

            numFramesRendered += 1;

            m_prevTime = currentTime;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Number of Frames Rendered= " << numFramesRendered << "\n";

    simulator->stopSimulation(m_physicalDevice);

    vkDeviceWaitIdle(m_logicalDevice);

    return EXIT_SUCCESS;
}
