#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <string>
#include <iostream>
#include <sstream>

#include <stdexcept>
#include <functional>
#include <vector>

namespace TestUtils {
    template<typename T>
    void assertEqual(T a, T b);

    template<typename T>
    void assertEqual(const std::vector<T>& expected, const std::vector<T>& actual);

    void assertTrue(bool b);

    void testRunner(const std::string& name, std::function<void()> fn);
}

template <typename T>
void TestUtils::assertEqual(T a, T b) {
    if (a != b) {
        std::stringstream ss;
        ss << "assertEqualsError: " << a << " does not equal " << b << "\n";
        throw std::runtime_error(ss.str());
    }
}

template <typename T>
void TestUtils::assertEqual(const std::vector<T>& expected, const std::vector<T>& actual) {
    assertEqual(expected.size(), actual.size());

    int numberOfErrors = 0;

    for (int i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            numberOfErrors += 1;

            std::cout << "Mismatch at index = " << i << " Expected = " << expected[i] << " Actual = " << actual[i] << "\n";
        }
    }

    assertTrue(numberOfErrors == 0);
}

#endif
