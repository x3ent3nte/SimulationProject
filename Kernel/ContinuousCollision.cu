#include "ContinuousCollision.cuh"

#include "InsertionSort.cuh"

namespace {

__device__
float agentMaxX(Agent& agent, float seconds) {
    float maxX = agent.position.x;

    if (agent.velocity.x > 0) {
        maxX += agent.velocity.x * seconds;
    }

    return maxX + agent.radius;
}

__device__
float agentMinX(Agent& agent, float seconds) {
    float minX = agent.position.x;

    if (agent.velocity.x < 0) {
        minX += agent.velocity.x * seconds;
    }

    return minX - agent.radius;
}

__global__
void mapAgentsToMaxXAndIndex(Agent* agents, MaxXAndIndex* maxXAndIndexes, int size, float seconds) {
  
    int tid = threadIdx.x;
    int globalOffset = blockDim.x * blockIdx.x;
    int gid = globalOffset + tid;

    if (gid >= size) { return; }

    Agent agent = agents[gid];
    
    maxXAndIndexes[gid] = {agentMaxX(agent, seconds), gid};
}

__global__
void mapAgentsToSortedIndex(Agent* in, Agent* out, MaxXAndIndex* maxXAndIndexes, int size) {

    int tid = threadIdx.x;
    int globalOffset = blockDim.x * blockIdx.x;
    int gid = globalOffset + tid;

    if (gid >= size) { return; }
    
    out[maxXAndIndexes[gid].index] = in[gid];
}

__global__
void detectCollisions(Agent* agents, int size, float seconds, CollisionAndTime* collisions, int* collisionFlags) {
    int tid = threadIdx.x;
    int globalOffset = blockDim.x * blockIdx.x;
    int gid = globalOffset + tid;

    if (gid >= size) { return; }

    Agent agent = agents[gid];
    
    float myMinX = agentMinX(agent, seconds);

    int collisionFlag = 0;
    CollisionAndTime collision = {0, 0, 0};

    for (int i = gid - 1; i >= 0; --i) {
        Agent other = agents[i];

        float otherMaxX = agentMaxX(other, seconds);

        if (otherMaxX < myMinX) {
            break;
        }
    }

    collisions[gid] = collision;
    collisionFlags[gid] = collisionFlag;
}

} // namespace anonymous

ContinuousCollision::ContinuousCollision(int maxAgents) {
    cudaMalloc(&m_maxXAndIndexes, maxAgents * sizeof(MaxXAndIndex));
    cudaMalloc(&m_needsSortingFlag, sizeof(int));
    cudaMalloc(&m_agentsBuffer, maxAgents * sizeof(Agent));
    cudaMalloc(&m_collisions, maxAgents * sizeof(CollisionAndTime));
    cudaMalloc(&m_collisionFlags, maxAgents * sizeof(int));
}

ContinuousCollision::~ContinuousCollision() {
    cudaFree(m_maxXAndIndexes);
    cudaFree(m_needsSortingFlag);
    cudaFree(m_agentsBuffer);
    cudaFree(m_collisions);
    cudaFree(m_collisionFlags);
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

    detectCollisions<<<numBlocks, kThreadsPerBlock>>>(m_agentsBuffer, size, seconds, m_collisions, m_collisionFlags);
}
