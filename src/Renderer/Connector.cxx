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
    VkCommandPool commandPool,
    VkQueue queue) {

    m_id = id;
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
    VkCommandPool commandPool,
    VkQueue queue,
    uint32_t numberOfElements) {

    size_t numBuffers = 3;

    m_logicalDevice = logicalDevice;

    m_buffers.resize(numBuffers);
    m_bufferMemories.resize(numBuffers);

    std::vector<AgentPositionAndRotation> initialPositions(numberOfElements);

    for (size_t i = 0; i < numberOfElements; ++i) {
        initialPositions[i] = AgentPositionAndRotation{glm::vec3(0.0f), glm::vec4(0.0f)};
    }

    for (size_t i = 0; i < numBuffers; ++i) {
        Buffer::createBufferWithData(
            initialPositions.data(),
            numberOfElements * sizeof(AgentPositionAndRotation),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            physicalDevice,
            logicalDevice,
            commandPool,
            queue,
            m_buffers[i],
            m_bufferMemories[i]);

        m_bufferIndexQueue.push_back(i);
    }

    m_newestBufferIndex = 0;

    m_newestConnectionId = 0;

    for (int i = 0; i < numBuffers; ++i) {
        auto connection = std::make_shared<Connection>(
            i,
            initialPositions.data(),
            initialPositions.size(),
            initialPositions.size() * sizeof(AgentPositionAndRotation),
            physicalDevice,
            logicalDevice,
            commandPool,
            queue);
        m_connections.push_back(connection);
    }
}

Connector::~Connector() {
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        vkFreeMemory(m_logicalDevice, m_bufferMemories[i], nullptr);
        vkDestroyBuffer(m_logicalDevice, m_buffers[i], nullptr);
    }
}

size_t Connector::takeNewestBufferIndex() {
    std::lock_guard<std::mutex> guard(m_mutex);
    size_t newest = m_bufferIndexQueue.front();
    m_bufferIndexQueue.pop_front();
    return newest;
}

size_t Connector::takeOldBufferIndex() {
    std::lock_guard<std::mutex> guard(m_mutex);
    size_t old = m_bufferIndexQueue.back();
    m_bufferIndexQueue.pop_back();
    return old;
}

void Connector::updateBufferIndex(size_t index) {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_newestBufferIndex = index;
    m_bufferIndexQueue.push_front(index);
}

void Connector::restoreBufferIndex(size_t index) {
    std::lock_guard<std::mutex> guard(m_mutex);
    if (index == m_newestBufferIndex) {
        m_bufferIndexQueue.push_front(index);
    } else {
        m_bufferIndexQueue.push_back(index);
    }
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
