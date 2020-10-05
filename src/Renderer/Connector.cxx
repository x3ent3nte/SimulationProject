#include <Renderer/Connector.h>

#include <Renderer/Buffer.h>
#include <Renderer/MyGLM.h>

#define NUM_ELEMENTS 32 * 512

Connector::Connector(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue queue) {

    size_t numBuffers = 3;

    m_buffers.resize(numBuffers);
    m_bufferMemories.resize(numBuffers);

    std::vector<glm::vec3> initialPositions(NUM_ELEMENTS);

    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        initialPositions[i] = glm::vec3(0.0f, 0.0f, 0.0f);
    }

    for (size_t i = 0; i < numBuffers; ++i) {
        Buffer::createReadOnlyBuffer(
            initialPositions.data(),
            NUM_ELEMENTS * sizeof(glm::vec3),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            physicalDevice,
            logicalDevice,
            commandPool,
            queue,
            m_buffers[i],
            m_bufferMemories[i]);

        m_bufferQueue.push_back(m_buffers[i]);
    }

    m_newestBuffer = m_buffers[0];
}

void Connector::cleanUp(VkDevice logicalDevice) {
    std::lock_guard<std::mutex> guard(m_mutex);
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        vkFreeMemory(logicalDevice, m_bufferMemories[i], nullptr);
        vkDestroyBuffer(logicalDevice, m_buffers[i], nullptr);
    }
}

VkBuffer Connector::takeNewest() {
    std::lock_guard<std::mutex> guard(m_mutex);
    VkBuffer newest = m_bufferQueue.front();
    m_bufferQueue.pop_front();
    return newest;
}

VkBuffer Connector::takeOld() {
    std::lock_guard<std::mutex> guard(m_mutex);
    VkBuffer old = m_bufferQueue.back();
    m_bufferQueue.pop_back();
    return old;
}

void Connector::updateBuffer(VkBuffer buffer) {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_newestBuffer = buffer;
    m_bufferQueue.push_front(buffer);
}

void Connector::restoreBuffer(VkBuffer buffer) {
    std::lock_guard<std::mutex> guard(m_mutex);
    if (buffer == m_newestBuffer) {
        m_bufferQueue.push_front(buffer);
    } else {
        m_bufferQueue.push_back(buffer);
    }
}
