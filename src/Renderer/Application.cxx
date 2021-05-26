#include <Renderer/Application.h>

#include <Renderer/Renderer.h>
#include <Renderer/Instance.h>
#include <Renderer/Surface.h>
#include <Renderer/PhysicalDevice.h>
#include <Renderer/LogicalDevice.h>
#include <Renderer/KeyboardControl.h>
#include <Renderer/Constants.h>
#include <Utils/Command.h>
#include <Renderer/Mesh.h>
#include <Renderer/Texture.h>

#include <Simulator/InputTerminal.h>
#include <Simulator/Simulator.h>

#include <Test/TestApplication.h>
#include <Utils/Timer.h>

#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <set>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <vector>

namespace {
    constexpr int kWidth = 1600;
    constexpr int kHeight = 1080;
} // anonymous namespace

class DefaultApplication : public Application {

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

    DefaultApplication() {

        m_keyboardControl = std::make_shared<KeyboardControl>();
        m_window = Surface::createWindow(m_keyboardControl, kWidth, kHeight);

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

        m_graphicsCommandPool = Command::createCommandPool(m_logicalDevice, indices.m_graphicsFamily);
        m_computeCommandPool = Command::createCommandPool(m_logicalDevice, indices.m_computeFamily);
    }

    ~DefaultApplication() {

        vkDestroyCommandPool(m_logicalDevice, m_graphicsCommandPool, nullptr);
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

    int run() override {

        {
            TestApplication(m_physicalDevice, m_logicalDevice, m_computeQueue, m_computeCommandPool).run();
        }

        const uint32_t maxNumberOfAgents = 64 * 512;
        const uint32_t maxNumberOfPlayers = 1;

        auto inputTerminal = std::make_shared<InputTerminal>();
        inputTerminal->addPlayer(m_keyboardControl);

        const std::vector<std::pair<std::string, std::string>> modelAndTexturePaths = {
            {Constants::kFreyjaModelPath, Constants::kFreyjaTexturePath},
            {Constants::kArwingModelPath, Constants::kArwingTexturePath},
            {Constants::kAsteroidModelPath, Constants::kAsteroidTexturePath},
            {Constants::kMoonModelPath, Constants::kMoonTexturePath}
        };

        std::vector<std::string> modelPaths(modelAndTexturePaths.size());
        std::vector<std::string> texturePaths(modelAndTexturePaths.size());

        for (size_t i = 0; i < modelAndTexturePaths.size(); ++i) {
            modelPaths[i] = modelAndTexturePaths[i].first;
            texturePaths[i] = modelAndTexturePaths[i].second;
        }

        auto mesh = std::make_shared<Mesh>(
            modelPaths,
            m_physicalDevice,
            m_logicalDevice,
            m_graphicsQueue,
            m_graphicsCommandPool);

        std::vector<std::shared_ptr<Texture>> textures(texturePaths.size());
        for (size_t i = 0; i < texturePaths.size(); ++i) {
            textures[i] = std::make_shared<Texture>(
                texturePaths[i],
                m_physicalDevice,
                m_logicalDevice,
                m_graphicsQueue,
                m_graphicsCommandPool);
        }

        auto connector = std::make_shared<Connector>(m_physicalDevice, m_logicalDevice, m_graphicsQueue, m_graphicsCommandPool, maxNumberOfAgents);
        auto simulator = std::make_shared<Simulator>(
            m_physicalDevice,
            m_logicalDevice,
            m_computeQueue,
            m_computeCommandPool,
            connector,
            inputTerminal,
            mesh,
            maxNumberOfAgents,
            maxNumberOfPlayers);
        simulator->simulate();

        std::shared_ptr<Renderer> renderer = Renderer::create(
            m_instance,
            m_window,
            m_surface,
            m_physicalDevice,
            m_logicalDevice,
            m_graphicsQueue,
            m_presentQueue,
            m_graphicsCommandPool,
            connector,
            mesh,
            textures,
            maxNumberOfAgents);

        m_prevTime = std::chrono::high_resolution_clock::now();

        int numFramesRendered = 0;
        try {
            while (!glfwWindowShouldClose(m_window->m_window)) {
                auto currentTime = std::chrono::high_resolution_clock::now();
                float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - m_prevTime).count();

                glfwPollEvents();
                //Timer timer("Render Frame " + numFramesRendered);
                renderer->render(time);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));

                numFramesRendered += 1;

                m_prevTime = currentTime;
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return EXIT_FAILURE;
        }

        simulator->stopSimulation(m_physicalDevice);

        std::cout << "Number of Frames Rendered = " << numFramesRendered << "\n";

        vkDeviceWaitIdle(m_logicalDevice);

        return EXIT_SUCCESS;
    }
};

std::shared_ptr<Application> Application::create() {
    return std::make_shared<DefaultApplication>();
}
