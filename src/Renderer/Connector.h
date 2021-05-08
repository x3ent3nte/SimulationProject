#ifndef CONNECTOR_H
#define CONNECTOR_H

#include <Simulator/Agent.h>

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

    const int m_id;

    std::vector<AgentPositionAndRotation> m_players;

    uint32_t m_numberOfElements;
    VkBuffer m_buffer;

    Connection(
        int id,
        VkDeviceSize memorySize,
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool);

    virtual ~Connection();
};

class Connector {

private:

    VkDevice m_logicalDevice;

    int m_newestConnectionId;

    std::mutex m_mutex;

    std::deque<std::shared_ptr<Connection>> m_connectionsQueue;

public:

    std::vector<std::shared_ptr<Connection>> m_connections;

    Connector(
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue queue,
        VkCommandPool commandPool,
        uint32_t numberOfElements);

    virtual ~Connector();

    std::shared_ptr<Connection> takeNewestConnection();
    std::shared_ptr<Connection> takeOldConnection();

    void restoreNewestConnection(std::shared_ptr<Connection> connection);
    void restoreConnection(std::shared_ptr<Connection> connection);
};

#endif
