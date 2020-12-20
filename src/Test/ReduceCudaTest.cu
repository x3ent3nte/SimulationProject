#include <Test/ReduceCudaTest.cuh>

#include <Cuda/Reduce.cuh>
#include <Utils/Timer.h>

namespace {

    __device__
    Collision earliestCollision(Collision a, Collision b) {
        if (a.time < b.time) {
            return a;
        } else {
            return b;
        }
    }
} // end namespace anonymous

Collision ReduceCudaTest::run(const std::vector<Collision>& data) {

    size_t bufferSize = data.size() * sizeof(Collision);

    Collision* d_collisionsOne;
    Collision* d_collisionsTwo;
    cudaMalloc(&d_collisionsOne, bufferSize);
    cudaMalloc(&d_collisionsTwo, bufferSize);

    cudaMemcpy(d_collisionsOne, data.data(), bufferSize, cudaMemcpyHostToDevice);

    Collision result;
    {
        Timer timer("Reduce CUDA");
        result = Reduce::reduce<Collision, earliestCollision>(d_collisionsOne, d_collisionsTwo, data.size());
    }

    cudaFree(d_collisionsOne);
    cudaFree(d_collisionsTwo);

    return result;
}
