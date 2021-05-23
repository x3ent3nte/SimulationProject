#include <Simulator/Impacter.h>

#include <Simulator/Agent.h>
#include <Utils/Buffer.h>
#include <Utils/Compute.h>
#include <Renderer/Command.h>

#include <vector>
#include <stdexcept>

namespace ImpacterUtil {

    constexpr size_t xDim = 1;

    constexpr size_t kNumberOfBindings = 2;

    VkDescriptorSetLayout createDescriptorSetLayout(VkDevice logicalDevice) {
        return Compute::createDescriptorSetLayout(logicalDevice, kNumberOfBindings);
    }

    VkDescriptorPool createDescriptorPool(VkDevice logicalDevice, size_t maxSets) {
        return Compute::createDescriptorPool(logicalDevice, kNumberOfBindings, maxSets);
    }

    VkDescriptorSet createDescriptorSet(
        VkDevice logicalDevice,
        VkDescriptorSetLayout descriptorSetLayout,
        VkDescriptorPool descriptorPool,
        VkBuffer agentsBuffer,
        VkBuffer collisionBuffer,
        uint32_t numberOfElements) {

        std::vector<Compute::BufferAndSize> bufferAndSizes = {
            {agentsBuffer, numberOfElements * sizeof(Agent)},
            {collisionBuffer, sizeof(Collision)}
        };

        return Compute::createDescriptorSet(
            logicalDevice,
            descriptorSetLayout,
            descriptorPool,
            bufferAndSizes);
    }

    VkPipeline createPipeline(
        VkDevice logicalDevice,
        VkPipelineLayout pipelineLayout) {

        return Compute::createPipeline("src/GLSL/spv/Impact.spv", logicalDevice, pipelineLayout);
    }

    VkCommandBuffer createCommandBuffer(
        VkDevice logicalDevice,
        VkCommandPool commandPool,
        VkPipeline pipeline,
        VkPipelineLayout pipelineLayout,
        VkDescriptorSet descriptorSet,
        VkBuffer collisionBuffer,
        VkBuffer collisionHostVisibleBuffer) {

        VkCommandBuffer commandBuffer;

        VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.commandPool = commandPool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(logicalDevice, &commandBufferAllocateInfo, &commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute command buffer");
        }

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin compute command buffer");
        }

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = sizeof(Collision);
        vkCmdCopyBuffer(commandBuffer, collisionHostVisibleBuffer, collisionBuffer, 1, &copyRegion);

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            0,
            nullptr);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

        uint32_t xGroups = 1;

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        vkCmdDispatch(commandBuffer, xGroups, 1, 1);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to end compute command buffer");
        }

        return commandBuffer;
    }
} // end anonymous namespace

Impacter::Impacter(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    VkBuffer agentsBuffer,
    uint32_t numberOfElements)
    : m_logicalDevice(logicalDevice)
    , m_queue(queue)
    , m_commandPool(commandPool) {

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        sizeof(Collision),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_collisionBuffer,
        m_collisionDeviceMemory);

    Buffer::createBuffer(
        physicalDevice,
        m_logicalDevice,
        sizeof(Collision),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_collisionHostVisibleBuffer,
        m_collisionHostVisibleDeviceMemory);

    m_descriptorSetLayout = ImpacterUtil::createDescriptorSetLayout(m_logicalDevice);
    m_descriptorPool = ImpacterUtil::createDescriptorPool(m_logicalDevice, 1);
    m_pipelineLayout = Compute::createPipelineLayout(m_logicalDevice, m_descriptorSetLayout);
    m_pipeline = ImpacterUtil::createPipeline(m_logicalDevice, m_pipelineLayout);
    m_descriptorSet = ImpacterUtil::createDescriptorSet(
        m_logicalDevice,
        m_descriptorSetLayout,
        m_descriptorPool,
        agentsBuffer,
        m_collisionBuffer,
        numberOfElements);

    m_commandBuffer = ImpacterUtil::createCommandBuffer(
        m_logicalDevice,
        m_commandPool,
        m_pipeline,
        m_pipelineLayout,
        m_descriptorSet,
        m_collisionBuffer,
        m_collisionHostVisibleBuffer);

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &m_fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute fence");
    }
}

Impacter::~Impacter() {
    vkFreeMemory(m_logicalDevice, m_collisionDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_collisionBuffer, nullptr);

    vkFreeMemory(m_logicalDevice, m_collisionHostVisibleDeviceMemory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_collisionHostVisibleBuffer, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_descriptorSetLayout, nullptr);

    vkDestroyDescriptorPool(m_logicalDevice, m_descriptorPool, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_pipeline, nullptr);

    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &m_commandBuffer);

    vkDestroyFence(m_logicalDevice, m_fence, nullptr);
}

void Impacter::run(const Collision& collision) {

    Collision collisionCopy = collision;
    Buffer::writeHostVisible(&collisionCopy, m_collisionHostVisibleDeviceMemory, 0, sizeof(Collision), m_logicalDevice);

    Command::runAndWait(m_commandBuffer, m_fence, m_queue, m_logicalDevice);
}
