#ifndef AGENT_H
#define AGENT_H

#include <Renderer/MyGLM.h>

struct Agent {
    uint32_t typeId;
    int32_t playerId;
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 acceleration;
    glm::vec3 target;
    glm::vec3 rotationalVelocity;
    glm::vec4 rotation;
    float radius;
    float mass;
};

struct AgentRenderInfo {
    uint32_t typeId;
    glm::vec3 position;
    glm::vec4 rotation;
};

#endif
