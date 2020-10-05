#include <Renderer/Renderer.h>

#include <Renderer/Simulator.h>
#include <Renderer/Utils.h>
#include <Renderer/Vertex.h>
#include <Renderer/Instance.h>
#include <Renderer/Surface.h>
#include <Renderer/PhysicalDevice.h>
#include <Renderer/LogicalDevice.h>
#include <Renderer/Constants.h>
#include <Renderer/Pipeline.h>
#include <Renderer/SwapChain.h>
#include <Renderer/Command.h>
#include <Renderer/Buffer.h>
#include <Renderer/Descriptors.h>
#include <Renderer/Image.h>
#include <Renderer/KeyboardControl.h>
#include <Renderer/MyMath.h>
#include <Renderer/MyGLM.h>
#include <Renderer/Connector.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

class HelloTriangleApplication {
public:
    void run() {
        m_keyboardControl = std::make_shared<KeyboardControl>();
        m_window = Surface::createWindow(m_keyboardControl);
        initVulkan();
        mainLoop();
        cleanUp();
    }

private:

    std::chrono::time_point<std::chrono::high_resolution_clock> m_prevTime;

    glm::vec3 m_cameraPosition;
    glm::vec3 m_cameraForward;
    glm::vec3 m_cameraUp;
    glm::vec3 m_cameraRight;

    std::shared_ptr<KeyboardControl> m_keyboardControl;
    std::shared_ptr<Surface::Window> m_window;

    VkInstance m_instance;
    VkDebugUtilsMessengerEXT m_debugMessenger;

    VkSurfaceKHR m_surface;

    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_logicalDevice;
    VkQueue m_graphicsQueue;
    VkQueue m_presentQueue;

    std::shared_ptr<Connector> m_connector;
    std::shared_ptr<Simulator> m_simulator;

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
    std::vector<VkCommandBuffer> m_commandBuffers;

    VkDescriptorPool m_descriptorPool;
    std::vector<VkDescriptorSet> m_descriptorSets;

    VkBuffer m_vertexBuffer;
    VkDeviceMemory m_vertexBufferMemory;
    VkBuffer m_indexBuffer;
    VkDeviceMemory m_indexBufferMemory;

    const size_t m_numberOfInstances = 32 * 512;
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

        m_instance = Instance::createInstance();
        Instance::setupDebugMessenger(m_instance, m_debugMessenger);
        m_surface = Surface::createSurface(m_instance, m_window->m_window);
        m_physicalDevice = PhysicalDevice::pickPhysicalDevice(m_instance, m_surface);
        m_msaaSamples = PhysicalDevice::getMaxUsableSampleCount(m_physicalDevice);
        LogicalDevice::createLogicalDevice(m_physicalDevice, m_surface, m_logicalDevice, m_graphicsQueue, m_presentQueue);

        m_simulator = std::make_shared<Simulator>(m_physicalDevice, m_logicalDevice);
        m_simulator->compute(m_logicalDevice);

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

        m_commandPool = Command::createCommandPool(m_physicalDevice, m_logicalDevice, m_surface);
        createColourResources();
        createDepthResources();
        createFrameBuffers();

        m_connector = std::make_shared<Connector>(m_physicalDevice, m_logicalDevice, m_commandPool, m_graphicsQueue);

        m_mipLevels = Image::createTextureImage(
            m_physicalDevice,
            m_logicalDevice,
            m_commandPool,
            m_graphicsQueue,
            m_textureImage,
            m_textureImageMemory);

        m_textureImageView = Image::createImageView(m_logicalDevice, m_textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, m_mipLevels);
        m_textureSampler = Image::createTextureSampler(m_logicalDevice, m_mipLevels);
        Utils::loadModel(m_vertices, m_indices, Constants::kModelPath);

        createInstanceBuffers();

        Buffer::createReadOnlyBuffer(
            m_vertices.data(),
            sizeof(m_vertices[0]) * m_vertices.size(),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            m_physicalDevice,
            m_logicalDevice,
            m_commandPool,
            m_graphicsQueue,
            m_vertexBuffer,
            m_vertexBufferMemory);

       Buffer::createReadOnlyBuffer(
            m_indices.data(),
            sizeof(m_indices[0]) * m_indices.size(),
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
    }

    void createInstanceBuffers() {
        m_instanceBuffers.resize(m_swapChainImages.size());
        m_instanceBufferMemories.resize(m_swapChainImages.size());

        std::vector<glm::vec3> instancePositions;
        instancePositions.resize(m_numberOfInstances);

        for (size_t i = 0; i < instancePositions.size(); ++i) {
            instancePositions[i] = MyMath::randomVec3InSphere(512.0f);
        }

        for (size_t i = 0; i < m_instanceBuffers.size(); ++i) {
            Buffer::createReadOnlyBuffer(
                instancePositions.data(),
                m_numberOfInstances * sizeof(glm::vec3),
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
        m_imageAvailableSemaphores.resize(Constants::kMaxFramesInFlight);
        m_renderFinishedSemaphores.resize(Constants::kMaxFramesInFlight);
        m_inFlightFences.resize(Constants::kMaxFramesInFlight);
        m_imagesInFlight.resize(m_swapChainImages.size(), VK_NULL_HANDLE);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < Constants::kMaxFramesInFlight; ++i) {
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

    void createCommandBuffers() {
        Command::createCommandBuffers(
            m_swapChainFrameBuffers,
            m_commandPool,
            m_logicalDevice,
            m_renderPass,
            m_swapChainExtent,
            m_instanceBuffers,
            m_vertexBuffer,
            m_indexBuffer,
            static_cast<uint32_t>(m_indices.size()),
            m_numberOfInstances,
            m_descriptorSets,
            m_graphicsPipeline,
            m_pipelineLayout,
            m_commandBuffers);
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

    void mainLoop() {
        m_prevTime = std::chrono::high_resolution_clock::now();
        while (!glfwWindowShouldClose(m_window->m_window)) {
            glfwPollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(m_logicalDevice);
    }

    void updateUniformBuffer(uint32_t currentImage) {

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - m_prevTime).count();

        KeyboardState keyboardState = m_keyboardControl->getKeyboardState();
        float delta = 10.0f * time;

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

        float angleDelta = 5.0f * time;

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

        //std::cout << "Camera x: " << m_cameraPosition.x << " y: " << m_cameraPosition.y << " z:" << m_cameraPosition.z << " Time: " << time << "\n";

        UniformBufferObject ubo{};
        //ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.model = glm::mat4(1.0f);
        ubo.view = glm::lookAt(m_cameraPosition, m_cameraPosition + m_cameraForward, m_cameraUp);
        ubo.proj = glm::perspective(glm::radians(45.0f), m_swapChainExtent.width / (float) m_swapChainExtent.height, 0.1f, 5000.f);

        ubo.proj[1][1] *= -1;

        void* data;
        vkMapMemory(m_logicalDevice, m_uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(m_logicalDevice, m_uniformBuffersMemory[currentImage]);

        m_prevTime = currentTime;
    }

    void drawFrame() {

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

        updateUniformBuffer(imageIndex);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {m_imageAvailableSemaphores[m_currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_commandBuffers[imageIndex];

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

        m_currentFrame = (m_currentFrame + 1) % Constants::kMaxFramesInFlight;
    }

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

        vkFreeCommandBuffers(m_logicalDevice, m_commandPool, static_cast<uint32_t>(m_commandBuffers.size()), m_commandBuffers.data());

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
        m_simulator->cleanUp(m_logicalDevice);
        m_connector->cleanUp(m_logicalDevice);

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

        for (size_t i = 0; i < Constants::kMaxFramesInFlight; ++i) {
            vkDestroySemaphore(m_logicalDevice, m_renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(m_logicalDevice, m_imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(m_logicalDevice, m_inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(m_logicalDevice, m_commandPool, nullptr);

        vkDestroyDevice(m_logicalDevice, nullptr);

        if(Constants::kEnableValidationLayers) {
            Instance::DestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        vkDestroyInstance(m_instance, nullptr);

        glfwDestroyWindow(m_window->m_window);
        glfwTerminate();
    }
};

int Renderer::render() {
    HelloTriangleApplication app;

    srand(time(NULL));
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
