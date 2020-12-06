#include <Simulator/Reduce.h>

#include <Simulator/ReduceUtil.h>

#include <math.h>

Reduce::Reduce(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t numberOfElements) {

    m_physicalDevice = physicalDevice;
    m_logicalDevice = logicalDevice;
    m_queue = queue;
    m_commandPool = commandPool;
}

Reduce::~Reduce() {

}

void Reduce::setDataSize(uint32_t dataSize) {

}

void Reduce::runReduceCommand(uint32_t dataSize) {
    setDataSize(dataSize);
}

VkBuffer Reduce::run(uint32_t dataSize) {

    VkBuffer bufferToReturn = m_bufferOne;
    VkBuffer otherBuffer = m_bufferTwo;

    while (dataSize > 1) {

        runReduceCommand(dataSize);

        dataSize = ceil(float(dataSize) / float(ReduceUtil::xDim * 2));

        VkBuffer temp = bufferToReturn;
        bufferToReturn = otherBuffer;
        otherBuffer = temp;
    }

    return bufferToReturn;
}
