#ifndef AGENT_H
#define AGENT_H

#include <Renderer/MyGLM.h>

struct Agent {
    glm::vec3 position;
    glm::vec3 target;
    glm::vec4 rotation;
};

struct AgentPositionAndRotation {
    glm::vec3 position;
    glm::vec4 rotation;
};

#endif
