#include <Simulator/Reducer.h>

#include <Simulator/Collision.h>
#include <Utils/Buffer.h>
#include <Utils/Compute.h>
#include <Utils/Command.h>

#include <array>
#include <stdexcept>

#include <math.h>

namespace ReducerUtil {

    constexpr size_t kXDim = 256;
    constexpr size_t kNumberOfBindings = 3;

    VkDescriptorSet createDescriptorSet(
        VkDevice logicalDevice,
        VkDescriptorSetLayout descriptorSetLayout,
        VkDescriptorPool descriptorPool,
        VkBuffer inBuffer,
        VkBuffer outBuffer,
        uint32_t numberOfElements) {

        std::vector<Compute::BufferAndSize> bufferAndSizes = {
            {inBuffer, numberOfElements * sizeof(Collision)},
            {outBuffer, numberOfElements * sizeof(Collision)}
        };

        return Compute::createDescriptorSet(
            logicalDevice,
            descriptorSetLayout,
            descriptorPool,
            bufferAndSizes);
    }

    bool createCommandBufferRecursive(
        VkCommandBuffer commandBuffer,
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkPipeline pipeline,
        VkPipelineLayout pipelineLayout,
        VkDescriptorSet descriptorSet,
        VkDescriptorSet otherDescriptorSet,
        uint32_t numberOfElements,
        bool switchBufferOutput) {

        if (numberOfElements > 1) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

            const uint32_t xGroups = ceil(((float) numberOfElements) / ((float) 2 * kXDim));

            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
            vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &numberOfElements);
            vkCmdDispatch(commandBuffer, xGroups, 1, 1);

            if (xGroups > 1) {

                VkMemoryBarrier memoryBarrier = {};
                memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                memoryBarrier.pNext = nullptr;
                memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
                memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

                vkCmdPipelineBarrier(
                    commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1,
                    &memoryBarrier,
                    0,
                    nullptr,
                    0,
                    nullptr);

                return createCommandBufferRecursive(
                    commandBuffer,
                    logicalDevice,
                    commandPool,
                    pipeline,
                    pipelineLayout,
                    otherDescriptorSet,
                    descriptorSet,
                    xGroups,
                    !switchBufferOutput);
            } else {
                return !switchBufferOutput;
            }
        }

        return switchBufferOutput;
    }
} // namespace ReducerUtil

Reducer::Reducer(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t numberOfElements) {

    m_physicalDevice = physicalDevice;
    m_logicalDevice = logicalDevice;
    m_queue = queue;
    m_commandPool = commandPool;

    Buffer::createBuffer(
        physicalDevice,
        logicalDevice,
        numberOfElements * sizeof(Collision),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_oneBuffer,
        m_oneBufferMemory);

    Buffer::createBuffer(
        physicalDevice,
        logicalDevice,
        numberOfElements * sizeof(Collision),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_twoBuffer,
        m_twoBufferMemory);

    m_descriptorSetLayout = Compute::createDescriptorSetLayout(m_logicalDevice, ReducerUtil::kNumberOfBindings);
    m_descriptorPool = Compute::createDescriptorPool(m_logicalDevice, ReducerUtil::kNumberOfBindings, 2);
    m_pipelineLayout = Compute::createPipelineLayoutWithPushConstant(m_logicalDevice, m_descriptorSetLayout, sizeof(uint32_t));
    m_pipeline = Compute::createPipeline("src/GLSL/spv/Reduce.spv", m_logicalDevice, m_pipelineLayout);

    m_oneToTwo = ReducerUtil::createDescriptorSet(
        m_logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        m_oneBuffer,
        m_twoBuffer,
        numberOfElements);

    m_twoToOne = ReducerUtil::createDescriptorSet(
        m_logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        m_twoBuffer,
        m_oneBuffer,
        numberOfElements);

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute fence");
    }

    m_currentNumberOfElements = numberOfElements;
    createCommandBuffer();
}

Reducer::~Reducer() {
    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &m_commandBuffer);

    vkFreeMemory(m_logicalDevice, m_oneBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_oneBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_twoBufferMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_twoBuffer, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_descriptorSetLayout, nullptr);

    vkDestroyDescriptorPool(m_logicalDevice, m_descriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_pipeline, nullptr);

    vkDestroyFence(m_logicalDevice, m_fence, nullptr);
}

void Reducer::createCommandBuffer() {

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = m_commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(m_logicalDevice, &commandBufferAllocateInfo, &m_commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute command buffer");
    }

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    if (vkBeginCommandBuffer(m_commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin compute command buffer");
    }

    bool switchBufferOutput = ReducerUtil::createCommandBufferRecursive(
        m_commandBuffer,
        m_logicalDevice,
        m_commandPool,
        m_pipeline,
        m_pipelineLayout,
        m_oneToTwo,
        m_twoToOne,
        m_currentNumberOfElements,
        false);

    m_returnBuffer = switchBufferOutput ? m_twoBuffer : m_oneBuffer;

    if (vkEndCommandBuffer(m_commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end compute command buffer");
    }
}

void Reducer::updateNumberOfElementsIfNecessary(uint32_t numberOfElements) {
    if (m_currentNumberOfElements != numberOfElements) {
        vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &m_commandBuffer);
        m_currentNumberOfElements = numberOfElements;
        createCommandBuffer();
    }
}

VkBuffer Reducer::run(uint32_t numberOfElements) {
    updateNumberOfElementsIfNecessary(numberOfElements);
    Command::runAndWait(m_commandBuffer, m_fence, m_queue, m_logicalDevice);
    return m_returnBuffer;
}
