#ifndef KEYBOARD_CONTROL_H
#define KEYBOARD_CONTROL_H

#include <mutex>

class InputState {

private:

    uint32_t m_state = 0;

public:

    bool isForward();
    bool isBack();
    bool isLeft();
    bool isRight();
    bool isUp();
    bool isDown();
    bool isPitchUp();
    bool isPitchDown();
    bool isYawLeft();
    bool isYawRight();
    bool isRollLeft();
    bool isRollRight();

    bool isBitSet(uint32_t bitMask);
    void setBitValue(bool set, uint32_t bitMask);
};

class KeyboardControl {

private:
    InputState m_inputState;

    std::mutex m_mutex;

public:

    InputState getInputState();

    void keyActivity(int key, int scancode, int action, int mods);
};

#endif
