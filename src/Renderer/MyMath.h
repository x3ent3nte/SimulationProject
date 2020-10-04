#ifndef MY_MATH_H
#define MY_MATH_H

#include <Renderer/MyGLM.h>

namespace MyMath {
    float cosineSimilarity(glm::vec3 a, glm::vec3 b);

    glm::vec4 hamiltonProduce(glm::vec4 a, glm::vec4 b);

    glm::vec4 createQuaternionFromAxisAndTheta(glm::vec3 axis, float theta);

    glm::vec4 inverseQuaternion(glm::vec4 q);

    glm::vec3 rotatePointByQuaternion(glm::vec3 p, glm::vec4 q);

    glm::vec3 rotatePointByAxisAndTheta(glm::vec3 p, glm::vec3 axis, float theta);

    float randomFloatBetweenZeroAndOne();

    float randomFloatBetweenMinusOneAndOne();

    glm::vec3 randomVec3InSphere(float radius);
}

#endif
