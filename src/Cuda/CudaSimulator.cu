#include <Cuda/CudaSimulator.cuh>

__device__
float magx(float3 v) {
    return sqrt((v.x * v.x) + (v.y * v.y) + (v.z * v.z));
}

__device__
float3 scalex(float3 v, float f) {
    return float3{v.x * f, v.y * f, v.z * f};
}

__device__
float3 addxx(float3 a, float3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__
float3 subxx(float3 a, float3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__
int hashInt(int a) {
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

__device__
float hashFloat(int a) {
    a = hashInt(a);
    int bound = (1 << 30) - 1;
    a &= bound;
    float b = float(a) / float(bound);
    return b;
}

__device__
float3 hashVec3(float radius, int seed) {
    const float z = (2.0f * hashFloat(seed)) - 1.0f;
    const float xyMag = sqrt(1.0f - (z * z));
    const float azimuth = hashFloat(seed + 1) * 6.28318530718;
    const float y = sin(azimuth) * xyMag;
    const float x = cos(azimuth) * xyMag;
    return scalex(float3{x, y, z}, radius * hashFloat(seed + 10));
}

__global__
void simulateKernel(CudaAgent* agents, float3* positions, size_t size) {
    int tid = threadIdx.x;
    int globalOffset = blockDim.x * blockIdx.x;
    int gid = globalOffset + tid;

    if (gid >= size) { return; }

    float maxDistance = 0.01;

    CudaAgent agent = agents[gid];

    for (size_t i = 0; i < 1; ++i) {
        float3 delta = subxx(agent.target, agent.position);
        float distanceBetweenTargetAndPosition = magx(delta);
        if (distanceBetweenTargetAndPosition < maxDistance) {
            agent.position = agent.target;
            agent.target = hashVec3(100.0, int(agent.position.x) + int(gid));
        } else {
            agent.position = addxx(agent.position, scalex(delta, maxDistance / distanceBetweenTargetAndPosition));
        }
    }

    agents[gid] = agent;
    positions[gid] = agent.position;
}

void CudaSimulator::simulate(CudaAgent* agents, float3* positions, size_t size) {
    const size_t threadsPerBlock = 512;
    for (size_t i = 0; i < 1; ++i) {
        simulateKernel<<<ceil((float) size / (float) threadsPerBlock), threadsPerBlock>>>(agents, positions, size);
    }
}
