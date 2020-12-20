#ifndef REDUCE_VULKAN_TEST_H
#define REDUCE_VULKAN_TEST_H

#include <Simulator/Reducer.h>
#include <Simulator/Collision.h>

#include <vulkan/vulkan.h>

#include <vector>
#include <memory>

class ReduceVulkanTest {

private:

    VkPhysicalDevice m_physicalDevice;
    VkDevice m_logicalDevice;
    VkQueue m_queue;
    VkCommandPool m_commandPool;

    std::shared_ptr<Reducer> m_reducer;

public:

ReduceVulkanTest(
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkQueue queue,
    VkCommandPool commandPool,
    uint32_t numberOfElements);

virtual ~ReduceVulkanTest();

Collision run(const std::vector<Collision>& data);

};

#endif
