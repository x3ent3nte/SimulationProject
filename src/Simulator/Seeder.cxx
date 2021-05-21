#include <Simulator/Seeder.h>

#include <Utils/MyMath.h>
#include <Renderer/MyGLM.h>

#include <math.h>

namespace {

Agent createSpaceShip(uint32_t typeId, float radius) {
    glm::vec3 position = MyMath::randomUnitVec3();
    position = (position * 300.0f) + (position * MyMath::randomFloatBetweenZeroAndOne() * 3000.0f);
    const glm::vec3 velocity = glm::vec3{0.0f, 0.0f, 0.0f};
    const glm::vec3 acceleration = glm::vec3(0.0f, 0.0f, 0.0f);
    const glm::vec3 target = MyMath::randomVec3InSphere(256.f) + position;
    const glm::vec4 rotation = MyMath::axisAndThetaToQuaternion(
        MyMath::randomVec3InSphere(1.0f),
        MyMath::randomFloatBetweenZeroAndOne() * MyMath::PI);
    const glm::vec3 rotationalVelocity = glm::vec3{0.0f, 0.0f, 0.0f};
    const float mass = 75;

    return Agent{typeId, -1, position, velocity, acceleration, target, rotationalVelocity, rotation, radius, mass};
}

Agent createFreyja(std::shared_ptr<Mesh> mesh) {
    const uint32_t typeId = 0;
    return createSpaceShip(typeId, mesh->m_subMeshInfos[typeId].radius);
}

Agent createArwing(std::shared_ptr<Mesh> mesh) {
    const uint32_t typeId = 1;
    return createSpaceShip(typeId, mesh->m_subMeshInfos[typeId].radius);
}

Agent createAsteroid(std::shared_ptr<Mesh> mesh) {
    const uint32_t typeId = 2;

    const float azimuth = MyMath::randomFloatBetweenMinusOneAndOne() * MyMath::PI;
    const glm::vec2 xzDir = {sin(azimuth), cos(azimuth)};
    const glm::vec2 xz = (xzDir * 5000.0f) + (MyMath::randomFloatBetweenZeroAndOne() * 3000.0f * xzDir);
    const float y = MyMath::randomFloatBetweenMinusOneAndOne() * 250;

    const glm::vec3 position = {xz.x, y, xz.y};
    const glm::vec3 velocity = glm::vec3{0.0f, 0.0f, 0.0f};
    const glm::vec3 acceleration = glm::vec3(0.0f, 0.0f, 0.0f);
    const glm::vec3 target = MyMath::randomVec3InSphere(256.f) + position;
    const glm::vec4 rotation = MyMath::axisAndThetaToQuaternion(
        MyMath::randomVec3InSphere(1.0f),
        MyMath::randomFloatBetweenZeroAndOne() * MyMath::PI);
    const glm::vec3 rotationalVelocity = glm::vec3{
        MyMath::randomFloatBetweenZeroAndOne() * MyMath::PI * 0.2,
        MyMath::randomFloatBetweenZeroAndOne() * MyMath::PI * 0.2,
        MyMath::randomFloatBetweenZeroAndOne() * MyMath::PI * 0.2};

    const float radius = mesh->m_subMeshInfos[typeId].radius;
    const float mass = 5000;
    return Agent{typeId, -1, position, velocity, acceleration, target, rotationalVelocity, rotation, radius, mass};
}

Agent createSun(std::shared_ptr<Mesh> mesh) {
    const uint32_t typeId = 3;

    const glm::vec3 position = {0.0f, 0.0f, 0.0f};
    const glm::vec3 velocity = glm::vec3{0.0f, 0.0f, 0.0f};
    const glm::vec3 acceleration = glm::vec3(0.0f, 0.0f, 0.0f);
    const glm::vec3 target = MyMath::randomVec3InSphere(256.f) + position;
    const glm::vec4 rotation = MyMath::axisAndThetaToQuaternion(
        MyMath::randomVec3InSphere(1.0f),
        MyMath::randomFloatBetweenZeroAndOne() * MyMath::PI);
    const glm::vec3 rotationalVelocity = glm::vec3{0.0f, 0.1f, 0.0f};

    const float radius = mesh->m_subMeshInfos[typeId].radius;
    const float mass = 99999999;
    return Agent{typeId, -1, position, velocity, acceleration, target, rotationalVelocity, rotation, radius, mass};
}

} // namespace anonymous

std::vector<Agent> Seeder::seed(
    uint32_t numberOfAgents,
    uint32_t numberOfPlayers,
    std::shared_ptr<Mesh> mesh) {

    const std::vector<Agent(*)(std::shared_ptr<Mesh>)> createFns = {createFreyja, createArwing, createAsteroid};

    std::vector<Agent> agents(numberOfAgents);
    for (size_t i = 0; i < numberOfAgents; ++i) {
        if (i == (numberOfAgents - 1)) {
            agents[i] = createSun(mesh);
        } else {
            agents[i] = createFns[i % createFns.size()](mesh);
        }
    }

    agents[0].playerId = 0;

    return agents;
}
