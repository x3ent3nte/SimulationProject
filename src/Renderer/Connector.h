#ifndef CONNECTOR_H
#define CONNECTOR_H

#include <vulkan/vulkan.h>

#include <deque>
#include <vector>
#include <mutex>

class Connector {

private:
    std::deque<VkBuffer> m_bufferQueue;
    VkBuffer m_newestBuffer;

    std::vector<VkBuffer> m_buffers;
    std::vector<VkDeviceMemory> m_bufferMemories;

    std::mutex m_mutex;

public:

    Connector(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue queue);
    virtual ~Connector() = default;

    void cleanUp(VkDevice logicalDevice);

    VkBuffer takeNewest();
    VkBuffer takeOld();

    void updateBuffer(VkBuffer buffer);
    void restoreBuffer(VkBuffer buffer);
};

#endif
