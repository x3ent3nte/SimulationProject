#include <Renderer/Renderer.h>

#include <Simulator/Agent.h>
#include <Utils/Utils.h>
#include <Renderer/Vertex.h>
#include <Renderer/Instance.h>
#include <Renderer/Surface.h>
#include <Renderer/PhysicalDevice.h>
#include <Renderer/LogicalDevice.h>
#include <Renderer/Constants.h>
#include <Renderer/Pipeline.h>
#include <Renderer/SwapChain.h>
#include <Renderer/Command.h>
#include <Utils/Buffer.h>
#include <Renderer/Descriptors.h>
#include <Renderer/Image.h>
#include <Renderer/KeyboardControl.h>
#include <Utils/MyMath.h>
#include <Renderer/MyGLM.h>

#include <Utils/Timer.h>

#include <vulkan/vulkan.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <thread>

namespace {
    constexpr int kMaxFramesInFlight = 2;
} // namespace anonymous

class DefaultRenderer : public Renderer {
public:
    DefaultRenderer(
        VkInstance instance,
        std::shared_ptr<Surface::Window> window,
        VkSurfaceKHR surface,
        std::shared_ptr<KeyboardControl> keyboardControl,
        VkPhysicalDevice physicalDevice,
        VkDevice logicalDevice,
        VkQueue graphicsQueue,
        VkQueue presentQueue,
        VkCommandPool commandPool,
        std::shared_ptr<Connector> connector,
        uint32_t maxNumberOfAgents)
        : m_maxNumberOfAgents(maxNumberOfAgents) {

        m_keyboardControl = keyboardControl;
        m_window = window;

        m_instance = instance;
        m_surface = surface;

        m_physicalDevice = physicalDevice;
        m_logicalDevice = logicalDevice;
        m_graphicsQueue = graphicsQueue;
        m_presentQueue = presentQueue;
        m_commandPool = commandPool;

        m_connector = connector;

        initVulkan();
    }

    virtual ~DefaultRenderer() {
        cleanUp();
    }

private:

    const uint32_t m_maxNumberOfAgents;

    glm::vec3 m_cameraPosition;
    glm::vec3 m_cameraForward;
    glm::vec3 m_cameraUp;
    glm::vec3 m_cameraRight;

    std::shared_ptr<KeyboardControl> m_keyboardControl;
    std::shared_ptr<Surface::Window> m_window;

    VkInstance m_instance;

    VkSurfaceKHR m_surface;

    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_logicalDevice;
    VkQueue m_graphicsQueue;
    VkQueue m_presentQueue;

    std::shared_ptr<Connector> m_connector;

    VkSwapchainKHR m_swapChain;
    std::vector<VkImage> m_swapChainImages;
    std::vector<VkImageView> m_swapChainImageViews;

    VkFormat m_swapChainImageFormat;
    VkExtent2D m_swapChainExtent;

    VkRenderPass m_renderPass;
    VkDescriptorSetLayout m_descriptorSetLayout;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_graphicsPipeline;

    std::vector<VkFramebuffer> m_swapChainFrameBuffers;

    VkCommandPool m_commandPool;
    std::vector<VkCommandBuffer> m_renderCommandBuffers;

    VkFence m_copyCompletedFence;

    VkDescriptorPool m_descriptorPool;
    std::vector<VkDescriptorSet> m_descriptorSets;

    VkBuffer m_vertexBuffer;
    VkDeviceMemory m_vertexBufferMemory;
    VkBuffer m_indexBuffer;
    VkDeviceMemory m_indexBufferMemory;

    std::vector<VkBuffer> m_instanceBuffers;
    std::vector<VkDeviceMemory> m_instanceBufferMemories;

    std::vector<VkBuffer> m_uniformBuffers;
    std::vector<VkDeviceMemory> m_uniformBuffersMemory;

    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    std::vector<VkFence> m_inFlightFences;
    std::vector<VkFence> m_imagesInFlight;
    size_t m_currentFrame = 0;

    VkImage m_textureImage;
    uint32_t m_mipLevels;
    VkDeviceMemory m_textureImageMemory;
    VkImageView m_textureImageView;
    VkSampler m_textureSampler;

    VkImage m_depthImage;
    VkDeviceMemory m_depthImageMemory;
    VkImageView m_depthImageView;

    std::vector<Vertex> m_vertices;
    std::vector<uint32_t> m_indices;

    VkSampleCountFlagBits m_msaaSamples = VK_SAMPLE_COUNT_1_BIT;

    VkImage m_colourImage;
    VkDeviceMemory m_colourImageMemory;
    VkImageView m_colourImageView;

    void recreateSwapChain() {
        std::cout << "recreateSwapChain called\n";

        int width = 0;
        int height = 0;

        glfwGetFramebufferSize(m_window->m_window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(m_window->m_window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(m_logicalDevice);

        cleanUpSwapChain();

        createSwapChain();

        m_renderPass = Pipeline::createRenderPass(
            m_logicalDevice,
            m_swapChainImageFormat,
            findDepthFormat(),
            m_msaaSamples);

        Pipeline::createPipeline(
            m_logicalDevice,
            m_swapChainExtent,
            m_msaaSamples,
            m_descriptorSetLayout,
            m_renderPass,
            m_pipelineLayout,
            m_graphicsPipeline);

        createColourResources();
        createDepthResources();
        createFrameBuffers();
        createUniformBuffers();

        m_descriptorPool = Descriptors::createDescriptorPool(
            m_logicalDevice,
            static_cast<uint32_t>(m_swapChainImages.size()));

        createDescriptorSets();
        createCommandBuffers();
    }

    void initVulkan() {
        m_cameraPosition = glm::vec3(2.0f, 0.0f, 1.0f);
        m_cameraForward = glm::vec3(-1.0f, 0.0f, 0.0f);
        m_cameraUp = glm::vec3(0.0f, 0.0f, 1.0f);
        m_cameraRight = glm::vec3(0.0f, 1.0f, 0.0f);

        m_msaaSamples = PhysicalDevice::getMaxUsableSampleCount(m_physicalDevice);

        createSwapChain();

        m_renderPass = Pipeline::createRenderPass(
            m_logicalDevice,
            m_swapChainImageFormat,
            findDepthFormat(),
            m_msaaSamples);

        m_descriptorSetLayout = Descriptors::createDescriptorSetLayout(m_logicalDevice);

        Pipeline::createPipeline(
            m_logicalDevice,
            m_swapChainExtent,
            m_msaaSamples,
            m_descriptorSetLayout,
            m_renderPass,
            m_pipelineLayout,
            m_graphicsPipeline);

        createColourResources();
        createDepthResources();
        createFrameBuffers();

        m_mipLevels = Image::createTextureImage(
            m_physicalDevice,
            m_logicalDevice,
            m_commandPool,
            m_graphicsQueue,
            m_textureImage,
            m_textureImageMemory);

        m_textureImageView = Image::createImageView(m_logicalDevice, m_textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, m_mipLevels);
        m_textureSampler = Image::createTextureSampler(m_logicalDevice, m_mipLevels);
        Utils::loadModel(m_vertices, m_indices, Constants::kFreyjaModelPath);

        createInstanceBuffers();

        Buffer::createBufferWithData(
            m_vertices.data(),
            sizeof(Vertex) * m_vertices.size(),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            m_physicalDevice,
            m_logicalDevice,
            m_commandPool,
            m_graphicsQueue,
            m_vertexBuffer,
            m_vertexBufferMemory);

       Buffer::createBufferWithData(
            m_indices.data(),
            sizeof(uint32_t) * m_indices.size(),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            m_physicalDevice,
            m_logicalDevice,
            m_commandPool,
            m_graphicsQueue,
            m_indexBuffer,
            m_indexBufferMemory);

        createUniformBuffers();

        m_descriptorPool = Descriptors::createDescriptorPool(
            m_logicalDevice,
            static_cast<uint32_t>(m_swapChainImages.size()));

        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(m_logicalDevice, &fenceInfo, nullptr, &m_copyCompletedFence);
    }

    void createInstanceBuffers() {
        m_instanceBuffers.resize(m_swapChainImages.size());
        m_instanceBufferMemories.resize(m_swapChainImages.size());

        std::vector<AgentPositionAndRotation> instancePositions;
        instancePositions.resize(m_maxNumberOfAgents);

        for (size_t i = 0; i < instancePositions.size(); ++i) {
            instancePositions[i] = AgentPositionAndRotation{
                MyMath::randomVec3InSphere(512.0f),
                MyMath::createQuaternionFromAxisAndTheta(glm::vec3(0.0f), 0.0f)};
        }

        for (size_t i = 0; i < m_instanceBuffers.size(); ++i) {
            Buffer::createBufferWithData(
                instancePositions.data(),
                m_maxNumberOfAgents * sizeof(AgentPositionAndRotation),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                m_physicalDevice,
                m_logicalDevice,
                m_commandPool,
                m_graphicsQueue,
                m_instanceBuffers[i],
                m_instanceBufferMemories[i]);
        }
    }

    void createDescriptorSets() {
        Descriptors::createDescriptorSets(
            m_logicalDevice,
            static_cast<uint32_t>(m_swapChainImages.size()),
            m_descriptorSetLayout,
            m_descriptorPool,
            m_uniformBuffers,
            m_textureImageView,
            m_textureSampler,
            m_descriptorSets);
    }

    void createColourResources() {
        VkFormat colourFormat = m_swapChainImageFormat;

        Image::createImage(
            m_physicalDevice,
            m_logicalDevice,
            m_swapChainExtent.width,
            m_swapChainExtent.height,
            1,
            m_msaaSamples,
            colourFormat,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            m_colourImage,
            m_colourImageMemory);

        m_colourImageView = Image::createImageView(m_logicalDevice, m_colourImage, colourFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }

    VkFormat findSupportedFormat(
        const std::vector<VkFormat>& candidates,
        VkImageTiling tiling,
        VkFormatFeatureFlags features) {

        for (VkFormat format : candidates) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(m_physicalDevice, format, &props);

            if ((tiling == VK_IMAGE_TILING_LINEAR) && ((props.linearTilingFeatures & features) == features)) {
                return format;
            } else if ((tiling == VK_IMAGE_TILING_OPTIMAL) && ((props.optimalTilingFeatures & features) == features)) {
                return format;
            }
        }

        throw std::runtime_error("Failed to find supported format");
    }

    VkFormat findDepthFormat() {
        return findSupportedFormat(
            {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }

    void createDepthResources() {
        VkFormat depthFormat = findDepthFormat();

        Image::createImage(
            m_physicalDevice,
            m_logicalDevice,
            m_swapChainExtent.width,
            m_swapChainExtent.height,
            1,
            m_msaaSamples,
            depthFormat,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            m_depthImage,
            m_depthImageMemory);

        m_depthImageView = Image::createImageView(m_logicalDevice, m_depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);

        Image::transitionImageLayout(
            m_logicalDevice,
            m_commandPool,
            m_graphicsQueue,
            m_depthImage,
            depthFormat,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            1);
    }

    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        m_uniformBuffers.resize(m_swapChainImages.size());
        m_uniformBuffersMemory.resize(m_swapChainImages.size());

        for (size_t i = 0; i < m_swapChainImages.size(); ++i) {
            Buffer::createBuffer(
                m_physicalDevice,
                m_logicalDevice,
                bufferSize,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_uniformBuffers[i],
                m_uniformBuffersMemory[i]);
        }
    }

    void createSyncObjects() {
        m_imageAvailableSemaphores.resize(kMaxFramesInFlight);
        m_renderFinishedSemaphores.resize(kMaxFramesInFlight);
        m_inFlightFences.resize(kMaxFramesInFlight);
        m_imagesInFlight.resize(m_swapChainImages.size(), VK_NULL_HANDLE);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < kMaxFramesInFlight; ++i) {
            if ((vkCreateSemaphore(m_logicalDevice, &semaphoreInfo, nullptr, &m_imageAvailableSemaphores[i]) != VK_SUCCESS) ||
                (vkCreateSemaphore(m_logicalDevice, &semaphoreInfo, nullptr, &m_renderFinishedSemaphores[i]) != VK_SUCCESS) ||
                (vkCreateFence(m_logicalDevice, &fenceInfo, nullptr, &m_inFlightFences[i]) != VK_SUCCESS)) {

                throw std::runtime_error("Failed to create semaphores");
            }
        }
    }

    void createFrameBuffers() {
        SwapChain::createFrameBuffers(
            m_logicalDevice,
            m_renderPass,
            m_swapChainExtent,
            m_colourImageView,
            m_depthImageView,
            m_swapChainImageViews,
            m_swapChainFrameBuffers);
    }

    void createSwapChain() {
        SwapChain::createSwapChain(
            m_physicalDevice,
            m_logicalDevice,
            m_surface,
            m_window->m_window,
            m_swapChainImageFormat,
            m_swapChainExtent,
            m_swapChain,
            m_swapChainImages,
            m_swapChainImageViews);
    }

    void updateUniformBuffer(uint32_t currentImage, float timeDelta) {

        KeyboardState keyboardState = m_keyboardControl->getKeyboardState();
        float delta = 100.0f * timeDelta;

        if (keyboardState.m_keyW) {
            m_cameraPosition += m_cameraForward * delta;
        }

        if (keyboardState.m_keyS) {
            m_cameraPosition -= m_cameraForward * delta;
        }

        if (keyboardState.m_keyA) {
            m_cameraPosition -= m_cameraRight * delta;
        }

        if (keyboardState.m_keyD) {
            m_cameraPosition += m_cameraRight * delta;
        }

        if (keyboardState.m_keyZ) {
            m_cameraPosition -= m_cameraUp * delta;
        }

        if (keyboardState.m_keyX) {
            m_cameraPosition += m_cameraUp * delta;
        }

        float angleDelta = 5.0f * timeDelta;

        if (keyboardState.m_keyQ) {
            m_cameraUp = MyMath::rotatePointByAxisAndTheta(m_cameraUp, m_cameraForward, -angleDelta);
            m_cameraRight = MyMath::rotatePointByAxisAndTheta(m_cameraRight, m_cameraForward, -angleDelta);
        }

        if (keyboardState.m_keyE) {
            m_cameraUp = MyMath::rotatePointByAxisAndTheta(m_cameraUp, m_cameraForward, angleDelta);
            m_cameraRight = MyMath::rotatePointByAxisAndTheta(m_cameraRight, m_cameraForward, angleDelta);
        }

        if (keyboardState.m_keyUp) {
            m_cameraForward = MyMath::rotatePointByAxisAndTheta(m_cameraForward, m_cameraRight, -angleDelta);
            m_cameraUp = MyMath::rotatePointByAxisAndTheta(m_cameraUp, m_cameraRight, -angleDelta);
        }

        if (keyboardState.m_keyDown) {
            m_cameraForward = MyMath::rotatePointByAxisAndTheta(m_cameraForward, m_cameraRight, angleDelta);
            m_cameraUp = MyMath::rotatePointByAxisAndTheta(m_cameraUp, m_cameraRight, angleDelta);
        }

        if (keyboardState.m_keyLeft) {
            m_cameraForward = MyMath::rotatePointByAxisAndTheta(m_cameraForward, m_cameraUp, angleDelta);
            m_cameraRight = MyMath::rotatePointByAxisAndTheta(m_cameraRight, m_cameraUp, angleDelta);
        }

        if (keyboardState.m_keyRight) {
            m_cameraForward = MyMath::rotatePointByAxisAndTheta(m_cameraForward, m_cameraUp, -angleDelta);
            m_cameraRight = MyMath::rotatePointByAxisAndTheta(m_cameraRight, m_cameraUp, -angleDelta);
        }

        //std::cout << "Camera x: " << m_cameraPosition.x << " y: " << m_cameraPosition.y << " z:" << m_cameraPosition.z << " Time: " << timeDelta << "\n";

        UniformBufferObject ubo{};
        //ubo.model = glm::rotate(glm::mat4(1.0f), timeDelta * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.model = glm::mat4(1.0f);
        ubo.view = glm::lookAt(m_cameraPosition, m_cameraPosition + m_cameraForward, m_cameraUp);
        ubo.proj = glm::perspective(glm::radians(45.0f), m_swapChainExtent.width / (float) m_swapChainExtent.height, 0.1f, 5000.f);
        ubo.cameraPosition = m_cameraPosition;

        ubo.proj[1][1] *= -1;

        void* data;
        vkMapMemory(m_logicalDevice, m_uniformBuffersMemory[currentImage], 0, sizeof(UniformBufferObject), 0, &data);
        memcpy(data, &ubo, sizeof(UniformBufferObject));
        vkUnmapMemory(m_logicalDevice, m_uniformBuffersMemory[currentImage]);
    }

    uint32_t updateAgentPositionsBuffer(size_t imageIndex) {
        Timer timer("XXXUpdate Agent Positions Buffer");

        auto connection = m_connector->takeNewestConnection();
        uint32_t numberOfElements = connection->m_numberOfElements;

        if (numberOfElements > 0) {
            VkCommandBuffer copyCommand = Buffer::recordCopyCommand(
                m_logicalDevice,
                m_commandPool,
                connection->m_buffer,
                m_instanceBuffers[imageIndex],
                sizeof(AgentPositionAndRotation) * numberOfElements);

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &copyCommand;
            vkResetFences(m_logicalDevice, 1, &m_copyCompletedFence);

            vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, m_copyCompletedFence);

            vkWaitForFences(m_logicalDevice, 1, &m_copyCompletedFence, VK_TRUE, UINT64_MAX);

            vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &copyCommand);
        }

        m_connector->restoreConnection(connection);

        return numberOfElements;
    }

    VkCommandBuffer createRenderCommand(size_t imageIndex, uint32_t numberOfInstances) {

        VkCommandBuffer commandBuffer;

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = m_commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(m_logicalDevice, &allocInfo, &commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffers");
        }

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin recording command buffer");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = m_renderPass;
        renderPassInfo.framebuffer = m_swapChainFrameBuffers[imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = m_swapChainExtent;

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {0.005f, 0.0f, 0.005f, 1.0f};
        clearValues[1].depthStencil = {1.0f, 0};

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);

        VkBuffer vertexBuffers[2] = {m_vertexBuffer, m_instanceBuffers[imageIndex]};
        VkDeviceSize offsets[] = {0, 0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);

        vkCmdBindIndexBuffer(commandBuffer, m_indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
            0, 1, &m_descriptorSets[imageIndex], 0, nullptr);

        vkCmdDrawIndexed(commandBuffer, m_indices.size(), numberOfInstances, 0, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer");
        }

        return commandBuffer;
    }

    void createCommandBuffers() {

        m_renderCommandBuffers = std::vector<VkCommandBuffer>();

        for (int i = 0; i < m_swapChainFrameBuffers.size(); ++i) {
            m_renderCommandBuffers.push_back(createRenderCommand(i, 0));
        }
    }

public:

    void render(float timeDelta) override {

        Timer timer("XXXXXX Render XXXXXX");

        vkWaitForFences(m_logicalDevice, 1, &m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        VkResult acquireImageResult = vkAcquireNextImageKHR(
            m_logicalDevice,
            m_swapChain,
            UINT64_MAX,
            m_imageAvailableSemaphores[m_currentFrame],
            VK_NULL_HANDLE,
            &imageIndex);

        if (acquireImageResult == VK_SUBOPTIMAL_KHR) {
            std::cout << "Suboptimal Swap Chain\n";
        }

        if (acquireImageResult == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        } else if (acquireImageResult != VK_SUCCESS && acquireImageResult != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("Failed to acquire swap chain image");
        }

        if (m_imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
            vkWaitForFences(m_logicalDevice, 1, &m_imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        }

        m_imagesInFlight[imageIndex] = m_inFlightFences[m_currentFrame];

        updateUniformBuffer(imageIndex, timeDelta);
        uint32_t numberOfElements = updateAgentPositionsBuffer(imageIndex);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {m_imageAvailableSemaphores[m_currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &m_renderCommandBuffers[imageIndex]);
        m_renderCommandBuffers[imageIndex] = createRenderCommand(imageIndex, numberOfElements);

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_renderCommandBuffers[imageIndex];

        VkSemaphore signalSemaphores[] = {m_renderFinishedSemaphores[m_currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        vkResetFences(m_logicalDevice, 1, &m_inFlightFences[m_currentFrame]);

        if (vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, m_inFlightFences[m_currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = {m_swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;

        VkResult presentResult = vkQueuePresentKHR(m_presentQueue, &presentInfo);

        if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR || m_window->m_hasBeenResized) {
            m_window->m_hasBeenResized = false;
            recreateSwapChain();
        } else if (presentResult != VK_SUCCESS) {
            throw std::runtime_error("Failed to present swap chain image");
        }

        m_currentFrame = (m_currentFrame + 1) % kMaxFramesInFlight;
    }

private:

    void cleanUpSwapChain() {
        vkDestroyImageView(m_logicalDevice, m_colourImageView, nullptr);
        vkDestroyImage(m_logicalDevice, m_colourImage, nullptr);
        vkFreeMemory(m_logicalDevice, m_colourImageMemory, nullptr);

        vkDestroyImageView(m_logicalDevice, m_depthImageView, nullptr);
        vkDestroyImage(m_logicalDevice, m_depthImage, nullptr);
        vkFreeMemory(m_logicalDevice, m_depthImageMemory, nullptr);

        for (auto frameBuffer : m_swapChainFrameBuffers) {
            vkDestroyFramebuffer(m_logicalDevice, frameBuffer, nullptr);
        }

        vkFreeCommandBuffers(m_logicalDevice, m_commandPool, static_cast<uint32_t>(m_renderCommandBuffers.size()), m_renderCommandBuffers.data());

        vkDestroyPipeline(m_logicalDevice, m_graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(m_logicalDevice, m_pipelineLayout, nullptr);
        vkDestroyRenderPass(m_logicalDevice, m_renderPass, nullptr);

        for (auto imageView : m_swapChainImageViews) {
            vkDestroyImageView(m_logicalDevice, imageView, nullptr);
        }

        vkDestroySwapchainKHR(m_logicalDevice, m_swapChain, nullptr);

        for (size_t i = 0; i < m_swapChainImages.size(); ++i) {
            vkDestroyBuffer(m_logicalDevice, m_uniformBuffers[i], nullptr);
            vkFreeMemory(m_logicalDevice, m_uniformBuffersMemory[i], nullptr);
        }

        vkDestroyDescriptorPool(m_logicalDevice, m_descriptorPool, nullptr);
    }

    void cleanUp() {

        cleanUpSwapChain();

        vkDestroySampler(m_logicalDevice, m_textureSampler, nullptr);
        vkDestroyImageView(m_logicalDevice, m_textureImageView, nullptr);

        vkDestroyImage(m_logicalDevice, m_textureImage, nullptr);
        vkFreeMemory(m_logicalDevice, m_textureImageMemory, nullptr);

        vkDestroyDescriptorSetLayout(m_logicalDevice, m_descriptorSetLayout, nullptr);

        vkDestroyBuffer(m_logicalDevice, m_indexBuffer, nullptr);
        vkFreeMemory(m_logicalDevice, m_indexBufferMemory, nullptr);

        vkDestroyBuffer(m_logicalDevice, m_vertexBuffer, nullptr);
        vkFreeMemory(m_logicalDevice, m_vertexBufferMemory, nullptr);

        const size_t numberOfBuffers = m_instanceBuffers.size();
        for (size_t i = 0; i < numberOfBuffers; ++i) {
            vkDestroyBuffer(m_logicalDevice, m_instanceBuffers[i], nullptr);
            vkFreeMemory(m_logicalDevice, m_instanceBufferMemories[i], nullptr);
        }

        for (size_t i = 0; i < kMaxFramesInFlight; ++i) {
            vkDestroySemaphore(m_logicalDevice, m_renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(m_logicalDevice, m_imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(m_logicalDevice, m_inFlightFences[i], nullptr);
        }
        vkDestroyFence(m_logicalDevice, m_copyCompletedFence, nullptr);
    }
};

std::shared_ptr<Renderer> Renderer::create(
    VkInstance instance,
    std::shared_ptr<Surface::Window> window,
    VkSurfaceKHR surface,
    std::shared_ptr<KeyboardControl> keyboardControl,
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue graphicsQueue,
    VkQueue presentQueue,
    VkCommandPool commandPool,
    std::shared_ptr<Connector> connector,
    uint32_t maxNumberOfAgents) {

    return std::make_shared<DefaultRenderer>(
        instance,
        window,
        surface,
        keyboardControl,
        physicalDevice,
        logicalDevice,
        graphicsQueue,
        presentQueue,
        commandPool,
        connector,
        maxNumberOfAgents);
}
