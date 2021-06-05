#include <Simulator/Seeder.h>

#include <Utils/MyMath.h>
#include <Utils/MyGLM.h>

#include <math.h>

namespace {

Agent createSpaceShip(std::shared_ptr<Mesh> mesh) {
    const uint32_t typeId = 0;
    glm::vec3 position = MyMath::randomUnitVec3();
    position = (position * 1600.0f) + (position * MyMath::randomFloatBetweenZeroAndOne() * 3000.0f);
    const glm::vec3 velocity = glm::vec3{0.0f, 0.0f, 0.0f};
    const glm::vec4 rotation = MyMath::axisAndThetaToQuaternion(
        MyMath::randomVec3InSphere(1.0f),
        MyMath::randomFloatBetweenZeroAndOne() * MyMath::PI);
    const glm::vec3 rotationalVelocity = glm::vec3{0.0f, 0.0f, 0.0f};
    const float mass = 120000;

    return Agent{typeId, -1, position, velocity, rotationalVelocity, rotation, mesh->m_subMeshInfos[typeId].radius, mass, 100.0f};
}

Agent createAsteroid(std::shared_ptr<Mesh> mesh) {
    const uint32_t typeId = 1;

    const float azimuth = MyMath::randomFloatBetweenMinusOneAndOne() * MyMath::PI;
    const glm::vec2 xzDir = {sin(azimuth), cos(azimuth)};
    const glm::vec2 xz = (xzDir * 2000.0f) + (MyMath::randomFloatBetweenZeroAndOne() * 3000.0f * xzDir);
    const float y = MyMath::randomFloatBetweenMinusOneAndOne() * 350;

    const glm::vec3 position = {xz.x, y, xz.y};
    const glm::vec3 velocityDir = MyMath::rotatePointByAxisAndTheta({xzDir.x, 0.0f, xzDir.y}, {0.0f, 1.0f, 0.0f}, MyMath::PI / 2);
    const glm::vec3 velocity = (velocityDir * 420.0f);
    const glm::vec4 rotation = MyMath::axisAndThetaToQuaternion(
        MyMath::randomVec3InSphere(1.0f),
        MyMath::randomFloatBetweenZeroAndOne() * MyMath::PI);
    const glm::vec3 rotationalVelocity = glm::vec3{
        MyMath::randomFloatBetweenZeroAndOne() * MyMath::PI * 0.2,
        MyMath::randomFloatBetweenZeroAndOne() * MyMath::PI * 0.2,
        MyMath::randomFloatBetweenZeroAndOne() * MyMath::PI * 0.2};

    const float radius = mesh->m_subMeshInfos[typeId].radius;
    const float mass = 7500000;
    return Agent{typeId, -1, position, velocity, rotationalVelocity, rotation, radius, mass, 100.0f};
}

Agent createSun(std::shared_ptr<Mesh> mesh) {
    const uint32_t typeId = 2;

    const glm::vec3 position = {0.0f, 0.0f, 0.0f};
    const glm::vec3 velocity = glm::vec3{0.0f, 0.0f, 0.0f};
    const glm::vec4 rotation = MyMath::axisAndThetaToQuaternion(
        MyMath::randomVec3InSphere(1.0f),
        MyMath::randomFloatBetweenZeroAndOne() * MyMath::PI);
    const glm::vec3 rotationalVelocity = glm::vec3{0.0f, 0.1f, 0.0f};

    const float radius = mesh->m_subMeshInfos[typeId].radius;
    const float mass = 6.417e18;
    return Agent{typeId, -1, position, velocity, rotationalVelocity, rotation, radius, mass, 1000000000.f};
}

Agent createPlayer(std::shared_ptr<Mesh> mesh) {
    Agent player = createSpaceShip(mesh);

    player.playerId = 0;
    player.position = {0.0f, 0.0f, 5000.0f};
    player.rotation = MyMath::axisAndThetaToQuaternion({0.0f, 1.0f, 0.0f}, 0);

    return player;
}

} // namespace anonymous

std::vector<Agent> Seeder::seed(
    uint32_t numberOfAgents,
    uint32_t numberOfPlayers,
    std::shared_ptr<Mesh> mesh) {

    std::vector<Agent> agents(numberOfAgents);
    for (size_t i = 0; i < numberOfAgents; ++i) {
        if (i == (numberOfAgents - 1)) {
            agents[i] = createSun(mesh);
        } else {
            int choice = rand() % 100;
            if (choice < 93) {
                agents[i] = createSpaceShip(mesh);
            } else {
                agents[i] =  createAsteroid(mesh);
            }
        }
    }

    agents[0] = createPlayer(mesh);

    return agents;
}
