#include "ContinuousCollision.cuh"

#include "InsertionSort.cuh"

namespace {

__global__
void mapAgentsToMaxXAndIndex(Agent* agents, MaxXAndIndex* maxXAndIndexes, int size, float seconds) {
  
    int tid = threadIdx.x;
    int globalOffset = blockDim.x * blockIdx.x;
    int gid = globalOffset + tid;

    if (gid >= size) { return; }

    Agent agent = agents[gid];

    float maxX;

    if (agent.velocity.x > 0) {
        maxX = agent.position.x * agent.velocity.x * seconds;
    } else {
        maxX = agent.position.x;
    }

    maxX += agent.radius;
    
    maxXAndIndexes[gid] = {maxX, gid};
}

__global__
void mapAgentsToSortedIndex(Agent* in, Agent* out, MaxXAndIndex* maxXAndIndexes, int size) {

    int tid = threadIdx.x;
    int globalOffset = blockDim.x * blockIdx.x;
    int gid = globalOffset + tid;

    if (gid >= size) { return; }

    
    out[maxXAndIndexes[gid].index] = in[gid];
}

} // namespace anonymous

ContinuousCollision::ContinuousCollision(int maxAgents) {
    cudaMalloc(&m_maxXAndIndexes, maxAgents * sizeof(MaxXAndIndex));
    cudaMalloc(&m_needsSortingFlag, sizeof(int));
    cudaMalloc(&m_agentsBuffer, maxAgents * sizeof(Agent));
}

ContinuousCollision::~ContinuousCollision() {
    cudaFree(m_maxXAndIndexes);
    cudaFree(m_needsSortingFlag);
    cudaFree(m_agentsBuffer);
}

__device__
int compareMaxXAndIndex(MaxXAndIndex a, MaxXAndIndex b) {
    return a.maxX > b.maxX;
}

void ContinuousCollision::collide(Agent* agents, int size, float seconds) {

    constexpr int kThreadsPerBlock = 256;

    int numBlocks = ceil(size / (float) kThreadsPerBlock);

    mapAgentsToMaxXAndIndex<<<numBlocks, kThreadsPerBlock>>>(agents, m_maxXAndIndexes, size, seconds);
    
    InsertionSort::sort<MaxXAndIndex, compareMaxXAndIndex>(m_maxXAndIndexes, m_needsSortingFlag, size);
    
    mapAgentsToSortedIndex<<<numBlocks, kThreadsPerBlock>>>(agents, m_agentsBuffer, m_maxXAndIndexes, size);
}
