#ifndef COLLISION_H
#define COLLISION_H

#include <cstdint>

struct Collision {
    uint32_t one;
    uint32_t two;
    float time;
};

struct ValueAndIndex {
    float value;
    uint32_t index;

    bool operator<(const ValueAndIndex& other) const;
};

#endif
