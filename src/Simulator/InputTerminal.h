#ifndef INPUT_TERMINAL_H
#define INPUT_TERMINAL_H

#include <Renderer/KeyboardControl.h>

#include <mutex>
#include <memory>
#include <vector>

class InputTerminal {

private:

    std::vector<std::shared_ptr<KeyboardControl>> m_players;

    std::mutex m_mutex;

public:

    virtual ~InputTerminal();

    void addPlayer(std::shared_ptr<KeyboardControl> player);

    std::vector<InputState> readInputStates();

};

#endif
