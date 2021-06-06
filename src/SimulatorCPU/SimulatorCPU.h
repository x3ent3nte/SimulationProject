#ifndef SIMULATOR_CPU_H
#define SIMULATOR_CPU_H

#include <memory>

class SimulatorCPU {

public:

    virtual ~SimulatorCPU() = default;

    virtual void start() = 0;

    virtual void stop() = 0;

    static std::shared_ptr<SimulatorCPU> create();
};

#endif
