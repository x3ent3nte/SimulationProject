#ifndef CONTINUOUS_COLLISION_CUH
#define CONTINUOUS_COLLISION_CUH

#include <Kernel/Agent.cuh>

typedef struct {
    float maxX;
    int index;
} MaxXAndIndex;

typedef struct {
    int one;
    int two;
    float seconds;
} CollisionAndTime;

class ContinuousCollision {

public:
    ContinuousCollision(int maxAgents);
    ~ContinuousCollision();
   
    void collide(Agent* agents, int size, float seconds);

private:
    MaxXAndIndex* m_maxXAndIndexes;
    int* m_needsSortingFlag;
    Agent* m_agentsBuffer;
    
    CollisionAndTime* m_collisions;
    CollisionAndTime* m_collisionsBuffer;
    
    int* m_collisionFlags;
    int* m_collisionFlagsOffsets;
};

#endif
