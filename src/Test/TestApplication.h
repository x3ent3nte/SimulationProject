#ifndef TEST_APPLICATION_H
#define TEST_APPLICATION_H

#include <vulkan/vulkan.h>

class TestApplication {

private:

    VkPhysicalDevice m_physicalDevice;
    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

public:

    TestApplication(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool);

    virtual ~TestApplication() = default;

    void run();

};

#endif
