#ifndef REDUCE_CUDA_TEST_H
#define REDUCE_CUDA_TEST_H

#include <Simulator/ReducerUtil.h>

#include <vector>

namespace ReduceCudaTest {

    ReducerUtil::Collision run(const std::vector<ReducerUtil::Collision>& data);
}

#endif
