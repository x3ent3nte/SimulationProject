#ifndef CUDA_SIMULATOR_H
#define CUDA_SIMULATOR_H

struct CudaAgent {
    float3 position;
    float3 target;
};

namespace CudaSimulator {
    void simulate(CudaAgent* agents, float3* position, size_t size);
}

#endif
