#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <string>
#include <iostream>
#include <sstream>

#include <stdexcept>

namespace TestUtils {
    template<typename T>
    void assertEqual(T a, T b);

    void assertTrue(bool b);
}

template <typename T>
void TestUtils::assertEqual(T a, T b) {
    if (a != b) {
        std::stringstream ss;
        ss << "assertEqualsError: " << a << " does not equal " << b << "\n";
        throw std::runtime_error(ss.str());
    }
}

#endif
