#ifndef COMPUTED_COLLISION_H
#define COMPUTED_COLLISION_H

#include <Utils/MyGLM.h>

#include <cstdint>

struct ComputedCollision {
    uint32_t agentIndex;
    float time;
    glm::vec3 velocityDelta;
};

#endif
