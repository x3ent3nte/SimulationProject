#ifndef CONNECTOR_H
#define CONNECTOR_H

#include <vulkan/vulkan.h>

#include <deque>
#include <vector>
#include <mutex>
#include <memory>

class Connection {

private:

    VkDevice m_logicalDevice;

    VkDeviceMemory m_deviceMemory;

public:

    int m_id;
    uint32_t m_numberOfElements;
    VkBuffer m_buffer;

    Connection(
        int id,
        void* initialState,
        size_t numberOfElements,
        size_t memorySize,
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkQueue queue);

    virtual ~Connection();

};

class Connector {

private:

    VkDevice m_logicalDevice;

    std::deque<std::shared_ptr<Connection>> m_connections;
    int m_newestConnectionId;

    std::deque<size_t> m_bufferIndexQueue;
    size_t m_newestBufferIndex;

    std::vector<VkDeviceMemory> m_bufferMemories;

    std::mutex m_mutex;

public:

    std::vector<VkBuffer> m_buffers;

    Connector(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkQueue queue,
        uint32_t numberOfElements);

    virtual ~Connector();

    size_t takeNewestBufferIndex();
    size_t takeOldBufferIndex();

    void updateBufferIndex(size_t index);
    void restoreBufferIndex(size_t index);

    std::shared_ptr<Connection> takeNewestConnection();
    std::shared_ptr<Connection> takeOldConnection();

    void restoreNewestConnection(std::shared_ptr<Connection> connection);
    void restoreConnection(std::shared_ptr<Connection> connection);
};

#endif
