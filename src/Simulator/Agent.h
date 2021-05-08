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
    glm::vec3 rotationalVelocity;
    glm::vec4 rotation;
    float radius;
};

struct AgentRenderInfo {
    int typeId;
    glm::vec3 position;
    glm::vec4 rotation;
};

#endif
