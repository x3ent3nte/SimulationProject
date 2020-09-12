#include <Renderer/KeyboardControl.h>

#include <iostream>

KeyboardState KeyboardControl::getKeyboardState() {
    std::lock_guard<std::mutex> guard(m_mutex);
    KeyboardState keyboardStateCopy =  m_keyboardState;
    return keyboardStateCopy;
}

void KeyboardControl::keyActivity(int key, int scancode, int action, int mods) {
    std::lock_guard<std::mutex> guard(m_mutex);

    bool valueToSet = (bool) action;
    switch (key) {
        case 87: { m_keyboardState.m_keyW = valueToSet; break; }

        case 65: { m_keyboardState.m_keyA = valueToSet; break; }

        case 83: { m_keyboardState.m_keyS = valueToSet; break; }

        case 68: { m_keyboardState.m_keyD = valueToSet; break; }

        case 81: { m_keyboardState.m_keyQ = valueToSet; break; }

        case 69: { m_keyboardState.m_keyE = valueToSet; break; }

        case 90: { m_keyboardState.m_keyZ = valueToSet; break; }

        case 88: { m_keyboardState.m_keyX = valueToSet; break; }

        case 265: { m_keyboardState.m_keyUp = valueToSet; break; }

        case 263: { m_keyboardState.m_keyLeft = valueToSet; break; }

        case 264: { m_keyboardState.m_keyDown = valueToSet; break; }

        case 262: { m_keyboardState.m_keyRight = valueToSet; break; }
    }
}
