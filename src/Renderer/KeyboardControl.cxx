#include <Renderer/KeyboardControl.h>

#include <iostream>

namespace {

    constexpr int kForwardBit = 1;
    constexpr int kBackBit = 2;
    constexpr int kLeftBit = 4;
    constexpr int kRightBit = 8;
    constexpr int kUpBit = 16;
    constexpr int kDownBit = 32;
    constexpr int kPitchUpBit = 64;
    constexpr int kPitchDownBit = 128;
    constexpr int kYawLeftBit = 256;
    constexpr int kYawRightBit = 512;
    constexpr int kRollLeftBit = 1024;
    constexpr int kRollRightBit = 2048;

} // namespace anonymous

bool InputState::isForward() { return isBitSet(kForwardBit); }
bool InputState::isBack() { return isBitSet(kBackBit); }
bool InputState::isLeft() { return isBitSet(kLeftBit); }
bool InputState::isRight() { return isBitSet(kRightBit); }
bool InputState::isUp() { return isBitSet(kUpBit); }
bool InputState::isDown() { return isBitSet(kDownBit); }
bool InputState::isPitchUp() { return isBitSet(kPitchUpBit); }
bool InputState::isPitchDown() { return isBitSet(kPitchDownBit); }
bool InputState::isYawLeft() { return isBitSet(kYawLeftBit); }
bool InputState::isYawRight() { return isBitSet(kYawRightBit); }
bool InputState::isRollLeft() { return isBitSet(kRollLeftBit); }
bool InputState::isRollRight() { return isBitSet(kRollRightBit); }

bool InputState::isBitSet(uint32_t bitMask) {
    return m_state & bitMask;
}

void InputState::setBitValue(bool set, uint32_t bitMask) {
    if (set) {
        m_state |= bitMask;
    } else {
        m_state &= (~bitMask);
    }
}


InputState KeyboardControl::readInputState() {
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_inputState;
}

void KeyboardControl::keyActivity(int key, int scancode, int action, int mods) {
    std::lock_guard<std::mutex> guard(m_mutex);

    bool valueToSet = (bool) action;
    switch (key) {
        case 87: { m_inputState.setBitValue(valueToSet, kForwardBit); break; }

        case 65: { m_inputState.setBitValue(valueToSet, kLeftBit); break; }

        case 83: { m_inputState.setBitValue(valueToSet, kBackBit); break; }

        case 68: { m_inputState.setBitValue(valueToSet, kRightBit); break; }

        case 81: { m_inputState.setBitValue(valueToSet, kRollLeftBit); break; }

        case 69: { m_inputState.setBitValue(valueToSet, kRollRightBit); break; }

        case 90: { m_inputState.setBitValue(valueToSet, kDownBit); break; }

        case 88: { m_inputState.setBitValue(valueToSet, kUpBit); break; }

        case 265: { m_inputState.setBitValue(valueToSet, kPitchDownBit); break; }

        case 263: { m_inputState.setBitValue(valueToSet, kYawLeftBit); break; }

        case 264: { m_inputState.setBitValue(valueToSet, kPitchUpBit); break; }

        case 262: { m_inputState.setBitValue(valueToSet, kYawRightBit); break; }
    }
}
