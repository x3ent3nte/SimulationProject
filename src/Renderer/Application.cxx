#include <Renderer/Application.h>

#include <Renderer/Instance.h>
#include <Renderer/Surface.h>
#include <Renderer/KeyboardControl.h>
#include <Renderer/Constants.h>

Application::Application() {

    m_keyboardControl = std::make_shared<KeyboardControl>();
    m_window = Surface::createWindow(m_keyboardControl);

    m_instance = Instance::createInstance();
    Instance::setupDebugMessenger(m_instance, m_debugMessenger);

    m_surface = Surface::createSurface(m_instance, m_window->m_window);

    m_renderer = std::make_shared<Renderer>();
}

Application::~Application() {

    if(Constants::kEnableValidationLayers) {
        Instance::DestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
    }

    vkDestroyInstance(m_instance, nullptr);
}

void Application::run() {
    m_renderer->render(m_instance, m_window, m_surface, m_keyboardControl);
}
