#include <Test/ReduceCudaTest.cuh>

#include <Cuda/Reduce.cuh>
#include <Utils/Timer.h>

namespace {

    __device__
    ReducerUtil::Collision earliestCollision(ReducerUtil::Collision a, ReducerUtil::Collision b) {
        if (a.time < b.time) {
            return a;
        } else {
            return b;
        }
    }
} // end namespace anonymous

ReducerUtil::Collision ReduceCudaTest::run(const std::vector<ReducerUtil::Collision>& data) {

    size_t bufferSize = data.size() * sizeof(ReducerUtil::Collision);

    ReducerUtil::Collision* d_collisionsOne;
    ReducerUtil::Collision* d_collisionsTwo;
    cudaMalloc(&d_collisionsOne, bufferSize);
    cudaMalloc(&d_collisionsTwo, bufferSize);

    cudaMemcpy(d_collisionsOne, data.data(), bufferSize, cudaMemcpyHostToDevice);

    ReducerUtil::Collision result;
    {
        Timer timer("Reduce CUDA");
        result = Reduce::reduce<ReducerUtil::Collision, earliestCollision>(d_collisionsOne, d_collisionsTwo, data.size());
    }

    cudaFree(d_collisionsOne);
    cudaFree(d_collisionsTwo);

    return result;
}
