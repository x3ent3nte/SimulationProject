#include "ContinuousCollision.cuh"

#include "InsertionSort.cuh"
#include "Scan.cuh"
#include "Reduce.cuh"
#include "MyMath.cuh"

namespace {

__device__
CollisionAndTime minCollisionAndTime(CollisionAndTime a, CollisionAndTime b) {
    if (a.seconds < b.seconds) {
        return a;
    } else {
        return b;
    }
}

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
float3 float3Scale(float3 a, float b) {
    return {a.x * b, a.y * b, a.z * b};
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

__device__
float distanceSquaredBetween(float3 a, float3 b) {
    float xd = a.x - b.x;
    float yd = a.y - b.y;
    float zd = a.z - b.z;

    return (xd * xd) + (yd * yd) + (zd * zd);
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
    CollisionAndTime collision = {-1, -1, seconds};

    for (int i = gid - 1; i >= 0; --i) {
        Agent other = agents[i];

        float otherMaxX = agentMaxX(other, seconds);

        if (otherMaxX < myMinX) {
            break;
        }

        float radiusSum = agent.radius + other.radius;
        float radiusSumSquared = radiusSum * radiusSum;

        // only collide if they are not already intersecting
        if (distanceSquaredBetween(agent.position, other.position) > radiusSumSquared) {
            
            float secondsOfCollision = calculateTimeOfCollision(agent, other);
            
            if (secondsOfCollision >= 0 && secondsOfCollision < seconds) {
                if (secondsOfCollision < collision.seconds) {
                    collisionFlag = 1;
                    collision = {gid, i, secondsOfCollision};
                }
            }
        }
    }

    collisions[gid] = collision;
    collisionFlags[gid] = collisionFlag;
}

__global__
void advanceTime(Agent* agents, int size, float seconds) {
    int tid = threadIdx.x;
    int globalOffset = blockDim.x * blockIdx.x;
    int gid = globalOffset + tid;

    if (gid >= size) { return; }

    Agent agent = agents[gid];
    agents[gid].position = float3Add(agent.position, float3Scale(agent.velocity, seconds));
}

} // namespace anonymous

ContinuousCollision::ContinuousCollision(int maxAgents) {
    cudaMalloc(&m_maxXAndIndexes, maxAgents * sizeof(MaxXAndIndex));
    cudaMalloc(&m_needsSortingFlag, sizeof(int));
    cudaMalloc(&m_agentsBuffer, maxAgents * sizeof(Agent));
    
    cudaMalloc(&m_collisions, maxAgents * sizeof(CollisionAndTime));
    cudaMalloc(&m_collisionsBuffer, maxAgents * sizeof(CollisionAndTime));
    
    cudaMalloc(&m_collisionFlags, maxAgents * sizeof(int));
    cudaMalloc(&m_collisionFlagsOffsets, maxAgents * sizeof(int));
}

ContinuousCollision::~ContinuousCollision() {
    cudaFree(m_maxXAndIndexes);
    cudaFree(m_needsSortingFlag);
    cudaFree(m_agentsBuffer);
    
    cudaFree(m_collisions);
    cudaFree(m_collisionsBuffer);
    
    cudaFree(m_collisionFlags);
    cudaFree(m_collisionFlagsOffsets);
}

__device__
int compareMaxXAndIndex(MaxXAndIndex a, MaxXAndIndex b) {
    return a.maxX > b.maxX;
}

__device__
int addx(int a, int b) {
    return a + b;
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

        CollisionAndTime earliestCollision = Reduce::reduce<CollisionAndTime, minCollisionAndTime>(m_collisions, m_collisionsBuffer, size);
        
        if (earliestCollision.one != -1) {
            advanceTime<<<numBlocks, kThreadsPerBlock>>>(m_agentsBuffer, size, earliestCollision.seconds);
            
            // TODO Resolve collision
            
            seconds -= earliestCollision.seconds;
            hadCollisions = 1;
        } else {
            advanceTime<<<numBlocks, kThreadsPerBlock>>>(m_agentsBuffer, size, seconds);
            hadCollisions = 0;
        }
    } while (hadCollisions);
}
