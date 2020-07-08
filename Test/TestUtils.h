#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <string>
#include <iostream>
#include <sstream>

namespace Test {
    template<typename T>
    void assertEquals(T a, T b);
}

template <typename T>
void Test::assertEquals(T a, T b) {
    if (a != b) {
        std::stringstream ss;
        ss << "assertEqualsError: " << a << " does not equal " << b >> "\n";
        throw ss.str();
    }
}

#endif
