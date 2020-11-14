#include <Test/TestUtils.h>

void TestUtils::assertTrue(bool b) {
    if (!b) {
        throw std::runtime_error("Expected True");
    }
}

void TestUtils::testRunner(const std::string& name, std::function<void()> fn) {
    try {
        std::cout << "\n[RUNNING " << name << "]\n\n";
        fn();
        std::cout << "\n[PASSED " << name << "]\n\n";
    } catch (const std::runtime_error& ex) {
        std::cout << "\n[FAILED " << name << "] " << ex.what() << "\n\n";
    }
}
