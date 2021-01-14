#include <Renderer/Command.h>

#include <Renderer/PhysicalDevice.h>

#include <array>
#include <stdexcept>

VkCommandPool Command::createCommandPool(
    VkDevice logicalDevice,
    uint32_t queueIndex) {

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueIndex;
    poolInfo.flags = 0;

    VkCommandPool commandPool;
    if (vkCreateCommandPool(logicalDevice, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }
    return commandPool;
}

VkCommandBuffer Command::beginSingleTimeCommands(VkDevice logicalDevice, VkCommandPool commandPool) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void Command::endSingleTimeCommands(
    VkCommandBuffer commandBuffer,
    VkQueue queue,
    VkDevice logicalDevice,
    VkCommandPool commandPool) {

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(logicalDevice, commandPool, 1, &commandBuffer);
}

void Command::createCommandBuffers(
    const std::vector<VkFramebuffer>& swapChainFrameBuffers,
    VkCommandPool commandPool,
    VkDevice logicalDevice,
    VkRenderPass renderPass,
    VkExtent2D swapChainExtent,
    const std::vector<VkBuffer>& instanceBuffers,
    VkBuffer vertexBuffer,
    VkBuffer indexBuffer,
    uint32_t numIndices,
    uint32_t numInstances,
    const std::vector<VkDescriptorSet>& descriptorSets,
    VkPipeline graphicsPipeline,
    VkPipelineLayout pipelineLayout,
    std::vector<VkCommandBuffer>& commandBuffers) {

    commandBuffers.resize(swapChainFrameBuffers.size());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

    if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers");
    }

    for (size_t i = 0; i < commandBuffers.size(); ++i) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;

        if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin recording command buffer");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFrameBuffers[i];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {0.005f, 0.0f, 0.005f, 1.0f};
        clearValues[1].depthStencil = {1.0f, 0};

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkBuffer vertexBuffers[2] = {vertexBuffer, instanceBuffers[i]};
        VkDeviceSize offsets[] = {0, 0};
        vkCmdBindVertexBuffers(commandBuffers[i], 0, 2, vertexBuffers, offsets);

        vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
            0, 1, &descriptorSets[i], 0, nullptr);

        vkCmdDrawIndexed(commandBuffers[i], numIndices, numInstances, 0, 0, 0);

        vkCmdEndRenderPass(commandBuffers[i]);

        if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer");
        }
    }
}
