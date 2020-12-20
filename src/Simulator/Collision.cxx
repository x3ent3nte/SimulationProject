#include <Simulator/Collision.h>

bool ValueAndIndex::operator<(const ValueAndIndex& other) const {
    return value < other.value;
}
