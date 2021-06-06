#include <SimulatorCPU/SimulatorCPU.h>

#include <atomic>
#include <thread>

class DefaultSimulatorCPU: public SimulatorCPU {

public:

    DefaultSimulatorCPU() {
        m_isActive = false;
    }

    ~DefaultSimulatorCPU() {

    }

    void simulate() {
        while (m_isActive) {

        }
    }

    void start() override {
        m_isActive = true;
        m_thread = std::thread(&DefaultSimulatorCPU::simulate, this);
    }

    void stop() override {
        m_isActive = false;
        m_thread.join();
    }

private:

    std::thread m_thread;
    std::atomic<bool> m_isActive;

};

std::shared_ptr<SimulatorCPU> SimulatorCPU::create() {
    return std::make_shared<DefaultSimulatorCPU>();
}
