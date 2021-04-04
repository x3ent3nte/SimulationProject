#include <Renderer/Connector.h>

#include <Utils/Buffer.h>
#include <Renderer/MyGLM.h>
#include <Simulator/Agent.h>


Connection::Connection(
    int id,
    void* initialState,
    size_t numberOfElements,
    size_t memorySize,
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool)
    : m_id(id) {

    m_numberOfElements = numberOfElements;

    m_logicalDevice = logicalDevice;

    Buffer::createBufferWithData(
        initialState,
        memorySize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        physicalDevice,
        m_logicalDevice,
        commandPool,
        queue,
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

    uint32_t numberOfBuffers = 3;

    m_logicalDevice = logicalDevice;

    std::vector<AgentPositionAndRotation> initialPositions(numberOfElements);

    for (size_t i = 0; i < numberOfElements; ++i) {
        initialPositions[i] = AgentPositionAndRotation{glm::vec3(0.0f), glm::vec4(0.0f)};
    }

    m_newestConnectionId = 0;

    for (int i = 0; i < numberOfBuffers; ++i) {
        auto connection = std::make_shared<Connection>(
            i,
            initialPositions.data(),
            initialPositions.size(),
            initialPositions.size() * sizeof(AgentPositionAndRotation),
            physicalDevice,
            logicalDevice,
            queue,
            commandPool);
        m_connections.push_back(connection);
    }
}

Connector::~Connector() {

}

std::shared_ptr<Connection> Connector::takeNewestConnection() {
    std::lock_guard<std::mutex> guard(m_mutex);
    auto newestConnection = m_connections.front();
    m_connections.pop_front();
    return newestConnection;
}

std::shared_ptr<Connection> Connector::takeOldConnection() {
    std::lock_guard<std::mutex> guard(m_mutex);
    auto oldConnection = m_connections.back();
    m_connections.pop_back();
    return oldConnection;
}

void Connector::restoreNewestConnection(std::shared_ptr<Connection> connection) {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_newestConnectionId = connection->m_id;
    m_connections.push_front(connection);
}

void Connector::restoreConnection(std::shared_ptr<Connection> connection) {
    std::lock_guard<std::mutex> guard(m_mutex);
    if (m_newestConnectionId == connection->m_id) {
        m_connections.push_front(connection);
    } else {
        m_connections.push_back(connection);
    }
}
