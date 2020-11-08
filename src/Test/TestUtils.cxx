#include <Test/TestUtils.h>

void TestUtils::assertTrue(bool b) {
    if (!b) {
        throw std::runtime_error("Expected True");
    }
}
