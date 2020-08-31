#ifndef PIPELINE_H
#define PIPELINE_H

#include <vulkan/vulkan.h>

namespace Pipeline {

    VkRenderPass createRenderPass(
        VkDevice device,
        VkFormat swapChainImageFormat,
        VkFormat depthFormat,
        VkSampleCountFlagBits msaaSamples);

    void createPipeline(
        VkPipelineLayout& pipelineLayout,
        VkPipeline& graphicsPipeline,
        VkDevice device,
        VkExtent2D swapChainExtent,
        VkSampleCountFlagBits msaaSamples,
        VkDescriptorSetLayout descriptorSetLayout,
        VkRenderPass renderPass);
}

#endif
