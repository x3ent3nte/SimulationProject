#include <Renderer/Connector.h>

#include <Renderer/Buffer.h>
#include <Renderer/MyGLM.h>
#include <Renderer/Constants.h>
#include <Renderer/Agent.h>
#include <Renderer/MyMath.h>

Connector::Connector(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue queue) {

    size_t numBuffers = 3;

    m_buffers.resize(numBuffers);
    m_bufferMemories.resize(numBuffers);

    std::vector<AgentPositionAndRotation> initialPositions(Constants::kNumberOfAgents);

    for (size_t i = 0; i < Constants::kNumberOfAgents; ++i) {
        initialPositions[i] = AgentPositionAndRotation{glm::vec3(0.0f), glm::vec4(0.0f)};
    }

    for (size_t i = 0; i < numBuffers; ++i) {
        Buffer::createReadOnlyBuffer(
            initialPositions.data(),
            Constants::kNumberOfAgents * sizeof(AgentPositionAndRotation),
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
}

void Connector::cleanUp(VkDevice logicalDevice) {
    std::lock_guard<std::mutex> guard(m_mutex);
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        vkFreeMemory(logicalDevice, m_bufferMemories[i], nullptr);
        vkDestroyBuffer(logicalDevice, m_buffers[i], nullptr);
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
