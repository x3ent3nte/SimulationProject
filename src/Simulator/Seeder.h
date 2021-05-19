#ifndef SEEDER_H
#define SEEDER_H

#include <Renderer/Mesh.h>

#include <Simulator/Agent.h>

#include <memory>
#include <vector>

namespace Seeder {

    std::vector<Agent> seed(
        uint32_t numberOfAgents,
        uint32_t numberOfPlayers,
        std::shared_ptr<Mesh> mesh);

}

#endif
