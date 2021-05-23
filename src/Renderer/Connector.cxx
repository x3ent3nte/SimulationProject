#include <Renderer/Connector.h>

#include <Utils/Buffer.h>
#include <Utils/MyGLM.h>
#include <Simulator/Agent.h>


Connection::Connection(
    int id,
    VkDeviceSize memorySize,
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool)
    : m_id(id) {

    m_numberOfElements = 0;

    m_logicalDevice = logicalDevice;

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        memorySize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_buffer,
        m_deviceMemory);
}

Connection::~Connection() {
    vkFreeMemory(m_logicalDevice, m_deviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_buffer, nullptr);
}

Connector::Connector(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t numberOfElements) {

    m_logicalDevice = logicalDevice;

    m_newestConnectionId = 0;

    int numberOfConnections = 3;

    for (uint32_t i = 0; i < numberOfConnections; ++i) {
        auto connection = std::make_shared<Connection>(
            i,
            numberOfElements * sizeof(AgentRenderInfo),
            physicalDevice,
            logicalDevice,
            queue,
            commandPool);
        m_connectionsQueue.push_back(connection);
        m_connections.push_back(connection);
    }
}

Connector::~Connector() {

}

std::shared_ptr<Connection> Connector::takeNewestConnection() {
    std::lock_guard<std::mutex> guard(m_mutex);
    auto newestConnection = m_connectionsQueue.front();
    m_connectionsQueue.pop_front();
    return newestConnection;
}

std::shared_ptr<Connection> Connector::takeOldConnection() {
    std::lock_guard<std::mutex> guard(m_mutex);
    auto oldConnection = m_connectionsQueue.back();
    m_connectionsQueue.pop_back();
    return oldConnection;
}

void Connector::restoreNewestConnection(std::shared_ptr<Connection> connection) {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_newestConnectionId = connection->m_id;
    m_connectionsQueue.push_front(connection);
}

void Connector::restoreConnection(std::shared_ptr<Connection> connection) {
    std::lock_guard<std::mutex> guard(m_mutex);
    if (m_newestConnectionId == connection->m_id) {
        m_connectionsQueue.push_front(connection);
    } else {
        m_connectionsQueue.push_back(connection);
    }
}
