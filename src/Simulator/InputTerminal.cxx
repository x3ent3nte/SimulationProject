#include <Simulator/InputTerminal.h>

InputTerminal::~InputTerminal() {

}

void InputTerminal::addPlayer(std::shared_ptr<KeyboardControl> player) {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_players.push_back(player);
}

std::vector<InputState> InputTerminal::readInputStates() {
    std::lock_guard<std::mutex> guard(m_mutex);

    std::vector<InputState> inputStates(m_players.size());

    for (int i = 0; i < m_players.size(); ++i) {
        inputStates[i] = m_players[i]->readInputState();
    }

    return inputStates;
}
