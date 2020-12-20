#ifndef REDUCE_CUDA_TEST_H
#define REDUCE_CUDA_TEST_H

#include <Simulator/Collision.h>

#include <vector>

namespace ReduceCudaTest {

    Collision run(const std::vector<Collision>& data);
}

#endif
