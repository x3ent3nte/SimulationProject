#ifndef KEYBOARD_CONTROL_H
#define KEYBOARD_CONTROL_H

#include <mutex>

struct KeyboardState {
    bool m_keyW = false;
    bool m_keyA = false;
    bool m_keyS = false;
    bool m_keyD = false;
    bool m_keyQ = false;
    bool m_keyE = false;
    bool m_keyZ = false;
    bool m_keyX = false;
    bool m_keyUp = false;
    bool m_keyLeft = false;
    bool m_keyDown = false;
    bool m_keyRight = false;
};

class KeyboardControl {

private:
    KeyboardState m_keyboardState;

    std::mutex m_mutex;

public:

    KeyboardState getKeyboardState();

    void keyActivity(int key, int scancode, int action, int mods);
};

#endif
