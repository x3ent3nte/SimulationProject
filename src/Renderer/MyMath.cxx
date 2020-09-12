#include <Renderer/MyMath.h>

float MyMath::cosineSimilarity(glm::vec3 a, glm::vec3 b) {
    float magMult = glm::length(a) * glm::length(b);
    if (magMult <= 0) {
        return 0;
    }

    return glm::dot(a, b) / magMult;
}

glm::vec4 MyMath::hamiltonProduce(glm::vec4 a, glm::vec4 b) {
    float r = (a.w * b.w) - (a.x * b.x) - (a.y * b.y) - (a.z * b.z);
    float i = (a.w * b.x) + (a.x * b.w) + (a.y * b.z) - (a.z * b.y);
    float j = (a.w * b.y) - (a.x * b.z) + (a.y * b.w) + (a.z * b.x);
    float k = (a.w * b.z) + (a.x * b.y) - (a.y * b.x) + (a.z * b.w);

    return glm::vec4(i, j, k, r);
}

glm::vec4 MyMath::createQuaternionFromAxisAndTheta(glm::vec3 axis, float theta) {
    float thetaHalved = theta / 2;
    return glm::vec4(sin(thetaHalved) * axis, cos(thetaHalved));
}

glm::vec4 MyMath::inverseQuaternion(glm::vec4 q) {
    return glm::vec4(-q.x, -q.y, -q.z, q.w);
}

glm::vec3 MyMath::rotatePointByQuaternion(glm::vec3 p, glm::vec4 q) {
    glm::vec4 p4 = glm::vec4(p, 0.0f);
    glm::vec4 qi = inverseQuaternion(q);

    return glm::vec3(hamiltonProduce(hamiltonProduce(q, p4), qi));
}

glm::vec3 MyMath::rotatePointByAxisAndTheta(glm::vec3 p, glm::vec3 axis, float theta) {
    return rotatePointByQuaternion(p, createQuaternionFromAxisAndTheta(axis, theta));
}
