#include "ContinuousCollision.cuh"

#include "InsertionSort.cuh"
#include "MyMath.cuh"

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

__device__
float3 quadraticFromFloat2(float2 a) {
    return {a.x * a.x, 2 * a.x * a.y, a.y * a.y};
}

__device__
float3 float3Add(float3 a, float3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__
float3 calculateQuadraticEquationOfCollision(Agent& one, Agent& two) {
    float radiusSum = one.radius + two.radius;
    float radiusSumSquared = radiusSum * radiusSum;

    float3 dxt = quadraticFromFloat2({two.velocity.x - one.velocity.x, two.position.x - one.position.x});
    float3 dyt = quadraticFromFloat2({two.velocity.y - one.velocity.y, two.position.y - one.position.y});
    float3 dzt = quadraticFromFloat2({two.velocity.z - one.velocity.z, two.position.z - one.position.z});

    float3 dt = float3Add(float3Add(dxt, dyt), dzt);
    dt.z -= radiusSumSquared;
    
    return dt;
}

typedef struct {
    int exists;
    float sol1;
    float sol2;
} QuadraticSolution;

__device__
QuadraticSolution solveQuadraticEquation(float3 q) {
    if (q.x == 0) {
        return {0, 0, 0};
    }

    float b2Minus4ac = (q.y * q.y) - (4 * q.x * q.z);

    if (b2Minus4ac < 0) {
        return {0, 0, 0};
    }

    float twoA = 2 * q.x;

    float sqrtB2Minus4Ac = sqrt(b2Minus4ac);

    float sol1 = (-q.y + sqrtB2Minus4Ac) / twoA;
    float sol2 = (-q.y - sqrtB2Minus4Ac) / twoA;

    return {1, sol1, sol2};
}

__device__
float calculateTimeOfCollision(Agent& one, Agent& two) {
    float3 q = calculateQuadraticEquationOfCollision(one, two);
    QuadraticSolution qSol = solveQuadraticEquation(q);
    
    if (qSol.exists) {
        if (qSol.sol1 < 0) {
            return qSol.sol2;
        } else {
            if (qSol.sol2 < 0) {
                return qSol.sol1;
            } else {
                return min(qSol.sol1, qSol.sol2);
            }
        }
    } else {
        return -1;
    }
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
    CollisionAndTime collision = {0, 0, seconds + 1};

    for (int i = gid - 1; i >= 0; --i) {
        Agent other = agents[i];

        float otherMaxX = agentMaxX(other, seconds);

        if (otherMaxX < myMinX) {
            break;
        }

        float secondsOfCollision = calculateTimeOfCollision(agent, other);

        if (secondsOfCollision >= 0 && secondsOfCollision < seconds) {
            if (secondsOfCollision < collision.seconds) {
                collisionFlag = 1;
                collision = {gid, i, secondsOfCollision};
            }
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

typedef struct {
    int hadCollisions;
    float seconds;
} HadCollisionAndSecondsEllapsed;

HadCollisionAndSecondsEllapsed resolveCollisions(Agent* agents, CollisionAndTime* collisions, int size, float seconds) {
    // TODO advance time to earliest collision and apply it
    return {1, 0.5};
}

void ContinuousCollision::collide(Agent* agents, int size, float seconds) {

    constexpr int kThreadsPerBlock = 256;

    int numBlocks = ceil(size / (float) kThreadsPerBlock);

    int hadCollisions = 0;

    do {

        mapAgentsToMaxXAndIndex<<<numBlocks, kThreadsPerBlock>>>(agents, m_maxXAndIndexes, size, seconds);
        
        InsertionSort::sort<MaxXAndIndex, compareMaxXAndIndex>(m_maxXAndIndexes, m_needsSortingFlag, size);
        
        mapAgentsToSortedIndex<<<numBlocks, kThreadsPerBlock>>>(agents, m_agentsBuffer, m_maxXAndIndexes, size);

        detectCollisions<<<numBlocks, kThreadsPerBlock>>>(m_agentsBuffer, size, seconds, m_collisions, m_collisionFlags);

        // TODO compact the collisions by scanning collisionFlags

        HadCollisionAndSecondsEllapsed result = resolveCollisions(m_agentsBuffer, m_collisions, size, seconds);

        seconds -= result.seconds;
        hadCollisions = result.hadCollisions;

    } while (hadCollisions);
}
