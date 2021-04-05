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

bool KeyboardState::isForward() { return isBitSet(kForwardBit); }
bool KeyboardState::isBack() { return isBitSet(kBackBit); }
bool KeyboardState::isLeft() { return isBitSet(kLeftBit); }
bool KeyboardState::isRight() { return isBitSet(kRightBit); }
bool KeyboardState::isUp() { return isBitSet(kUpBit); }
bool KeyboardState::isDown() { return isBitSet(kDownBit); }
bool KeyboardState::isPitchUp() { return isBitSet(kPitchUpBit); }
bool KeyboardState::isPitchDown() { return isBitSet(kPitchDownBit); }
bool KeyboardState::isYawLeft() { return isBitSet(kYawLeftBit); }
bool KeyboardState::isYawRight() { return isBitSet(kYawRightBit); }
bool KeyboardState::isRollLeft() { return isBitSet(kRollLeftBit); }
bool KeyboardState::isRollRight() { return isBitSet(kRollRightBit); }

bool KeyboardState::isBitSet(uint32_t bitMask) {
    return m_state & bitMask;
}

void KeyboardState::setBitValue(bool set, uint32_t bitMask) {
    if (set) {
        m_state |= bitMask;
    } else {
        m_state &= (~bitMask);
    }
}


KeyboardState KeyboardControl::getKeyboardState() {
    std::lock_guard<std::mutex> guard(m_mutex);
    KeyboardState keyboardStateCopy =  m_keyboardState;
    return keyboardStateCopy;
}

void KeyboardControl::keyActivity(int key, int scancode, int action, int mods) {
    std::lock_guard<std::mutex> guard(m_mutex);

    bool valueToSet = (bool) action;
    switch (key) {
        case 87: { m_keyboardState.setBitValue(valueToSet, kForwardBit); break; }

        case 65: { m_keyboardState.setBitValue(valueToSet, kLeftBit); break; }

        case 83: { m_keyboardState.setBitValue(valueToSet, kBackBit); break; }

        case 68: { m_keyboardState.setBitValue(valueToSet, kRightBit); break; }

        case 81: { m_keyboardState.setBitValue(valueToSet, kRollLeftBit); break; }

        case 69: { m_keyboardState.setBitValue(valueToSet, kRollRightBit); break; }

        case 90: { m_keyboardState.setBitValue(valueToSet, kDownBit); break; }

        case 88: { m_keyboardState.setBitValue(valueToSet, kUpBit); break; }

        case 265: { m_keyboardState.setBitValue(valueToSet, kPitchDownBit); break; }

        case 263: { m_keyboardState.setBitValue(valueToSet, kYawLeftBit); break; }

        case 264: { m_keyboardState.setBitValue(valueToSet, kPitchUpBit); break; }

        case 262: { m_keyboardState.setBitValue(valueToSet, kYawRightBit); break; }
    }
}
