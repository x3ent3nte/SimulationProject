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
        VkDevice device,
        VkExtent2D swapChainExtent,
        VkSampleCountFlagBits msaaSamples,
        VkDescriptorSetLayout descriptorSetLayout,
        VkRenderPass renderPass,
        VkPipelineLayout& pipelineLayout,
        VkPipeline& graphicsPipeline);
}

#endif
