#ifndef REDUCE_VULKAN_TEST_H
#define REDUCE_VULKAN_TEST_H

#include <Simulator/Reducer.h>
#include <Simulator/ReducerUtil.h>

#include <vector>

class ReduceVulkanTest {

private:

public:

virtual ~ReduceVulkanTest();

ReducerUtil::Collision run(const std::vector<ReducerUtil::Collision>& data);

};

#endif
