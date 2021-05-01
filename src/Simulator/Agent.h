#ifndef AGENT_H
#define AGENT_H

#include <Renderer/MyGLM.h>

struct Agent {
    int typeId;
    int playerId;
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 acceleration;
    glm::vec3 target;
    glm::vec4 rotation;
    glm::vec4 rotationalVelocity;
    float radius;
};

struct AgentPositionAndRotation {
    glm::vec3 position;
    glm::vec4 rotation;
};

#endif
