#ifndef CONNECTOR_H
#define CONNECTOR_H

#include <vulkan/vulkan.h>

#include <deque>
#include <vector>
#include <mutex>

class Connector {

public:

    std::deque<size_t> m_bufferIndexQueue;
    size_t m_newestBufferIndex;

    std::vector<VkBuffer> m_buffers;
    std::vector<VkDeviceMemory> m_bufferMemories;

    std::mutex m_mutex;

    Connector(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue queue);
    virtual ~Connector() = default;

    void cleanUp(VkDevice logicalDevice);

    size_t takeNewestBufferIndex();
    size_t takeOldBufferIndex();

    void updateBufferIndex(size_t index);
    void restoreBufferIndex(size_t index);
};

#endif
