#ifndef APPLICATION_H
#define APPLICATION_H

#include <Renderer/Renderer.h>

#include <vulkan/vulkan.h>

#include <memory>

class Application {

private:

    std::shared_ptr<KeyboardControl> m_keyboardControl;
    std::shared_ptr<Surface::Window> m_window;

    VkInstance m_instance;
    VkDebugUtilsMessengerEXT m_debugMessenger;

    VkSurfaceKHR m_surface;

    std::shared_ptr<Renderer> m_renderer;

public:

    Application();

    virtual ~Application();

    void run();
};

#endif
