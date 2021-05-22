#include <Utils/Buffer.h>

#include <Renderer/PhysicalDevice.h>
#include <Renderer/Command.h>

#include <stdexcept>

// Record copy command buffer at specified offsets, so ideally should be reused
VkCommandBuffer Buffer::recordCopyCommand(
    VkDevice logicalDevice,
    VkCommandPool commandPool,
    VkBuffer srcBuffer,
    VkBuffer dstBuffer,
    VkDeviceSize size,
    size_t srcOffset,
    size_t dstOffset) {

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = srcOffset;
    copyRegion.dstOffset = dstOffset;
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    return commandBuffer;
}

// Record copy command buffer, so ideally should be reused
VkCommandBuffer Buffer::recordCopyCommand(
    VkDevice logicalDevice,
    VkCommandPool commandPool,
    VkBuffer srcBuffer,
    VkBuffer dstBuffer,
    VkDeviceSize size) {

    return recordCopyCommand(
        logicalDevice,
        commandPool,
        srcBuffer,
        dstBuffer,
        size,
        0,
        0);
}

// Perform one time copy at specified offsets
void Buffer::copyBuffer(
    VkDevice logicalDevice,
    VkCommandPool commandPool,
    VkQueue queue,
    VkBuffer& srcBuffer,
    VkBuffer& dstBuffer,
    VkDeviceSize size,
    size_t srcOffset,
    size_t dstOffset) {

    VkCommandBuffer commandBuffer = Command::beginSingleTimeCommands(logicalDevice, commandPool);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = srcOffset;
    copyRegion.dstOffset = dstOffset;
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    Command::endSingleTimeCommands(commandBuffer, queue, logicalDevice, commandPool);
}

// Perform one time copy
void Buffer::copyBuffer(
    VkDevice logicalDevice,
    VkCommandPool commandPool,
    VkQueue queue,
    VkBuffer& srcBuffer,
    VkBuffer& dstBuffer,
    VkDeviceSize size) {

    copyBuffer(
        logicalDevice,
        commandPool,
        queue,
        srcBuffer,
        dstBuffer,
        size,
        0,
        0);
}

void Buffer::createBuffer(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkBuffer& buffer,
    VkDeviceMemory& bufferMemory) {

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(logicalDevice, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create vertex buffer");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(logicalDevice, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = PhysicalDevice::findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate vertex buffer memory");
    }

    if (vkBindBufferMemory(logicalDevice, buffer, bufferMemory, 0) != VK_SUCCESS) {
        throw std::runtime_error("Failed to bind buffer");
    }
}

void Buffer::createBufferWithData(
    void* data,
    VkDeviceSize bufferSize,
    VkBufferUsageFlags usage,
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkCommandPool commandPool,
    VkQueue queue,
    VkBuffer& buffer,
    VkDeviceMemory& bufferMemory) {

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    Buffer::createBuffer(
        physicalDevice,
        logicalDevice,
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer,
        stagingBufferMemory);

    Buffer::writeHostVisible(data, stagingBufferMemory, 0, bufferSize, logicalDevice);

    Buffer::createBuffer(
        physicalDevice,
        logicalDevice,
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        buffer,
        bufferMemory);

    Buffer::copyBuffer(logicalDevice, commandPool, queue, stagingBuffer, buffer, bufferSize);

    vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
}

void Buffer::writeHostVisible(
    void* src,
    VkDeviceMemory dst,
    VkDeviceSize offset,
    VkDeviceSize size,
    VkDevice logicalDevice) {

    void* dataMap;
    vkMapMemory(logicalDevice, dst, offset, size, 0, &dataMap);
    memcpy(dataMap, src, size);
    vkUnmapMemory(logicalDevice, dst);
}

void Buffer::readHostVisible(
    VkDeviceMemory src,
    void* dst,
    VkDeviceSize offset,
    VkDeviceSize size,
    VkDevice logicalDevice) {

    void* dataMap;
    vkMapMemory(logicalDevice, src, offset, size, 0, &dataMap);
    memcpy(dst, dataMap, size);
    vkUnmapMemory(logicalDevice, src);
}

// Copy device to host. Creates temporary buffers each time, so not very optimal
void Buffer::copyDeviceBufferToHost(
    void* data,
    VkDeviceSize bufferSize,
    VkBuffer buffer,
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkCommandPool commandPool,
    VkQueue queue) {

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    Buffer::createBuffer(
        physicalDevice,
        logicalDevice,
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer,
        stagingBufferMemory);

    Buffer::copyBuffer(logicalDevice, commandPool, queue, buffer, stagingBuffer, bufferSize);

    Buffer::readHostVisible(stagingBufferMemory, data, 0, bufferSize, logicalDevice);

    vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
}

// Copy host to device. Creates temporary buffers each time, so not very optimal
void Buffer::copyHostToDeviceBuffer(
    void* data,
    VkDeviceSize bufferSize,
    VkBuffer buffer,
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkCommandPool commandPool,
    VkQueue queue) {

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    Buffer::createBuffer(
        physicalDevice,
        logicalDevice,
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer,
        stagingBufferMemory);

    Buffer::writeHostVisible(data, stagingBufferMemory, 0, bufferSize, logicalDevice);

    Buffer::copyBuffer(logicalDevice, commandPool, queue, stagingBuffer, buffer, bufferSize);

    vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
}

// Used by image renderer
void Buffer::copyBufferToImage(
    VkDevice logicalDevice,
    VkCommandPool commandPool,
    VkQueue queue,
    VkBuffer buffer,
    VkImage image,
    uint32_t width,
    uint32_t height) {

    VkCommandBuffer commandBuffer = Command::beginSingleTimeCommands(logicalDevice, commandPool);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(
        commandBuffer, buffer, image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    Command::endSingleTimeCommands(commandBuffer, queue, logicalDevice, commandPool);
}
