#ifndef CONTINUOUS_COLLISION_CUH
#define CONTINUOUS_COLLISION_CUH

#include "Agent.cuh"

typedef struct {
    float maxX;
    int index;
} MaxXAndIndex;

class ContinuousCollision {

public:
    ContinuousCollision(int maxAgents);
    ~ContinuousCollision();
   
    void collide(Agent* agents, int size, float seconds);

private:
    MaxXAndIndex* m_maxXAndIndexes;
    int* m_needsSortingFlag;
    Agent* m_agentsBuffer;
};

#endif
