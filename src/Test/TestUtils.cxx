#include <Test/TestUtils.h>

void TestUtils::assertTrue(bool b) {
    if (!b) {
        throw std::runtime_error("Expected True");
    }
}

void TestUtils::testRunner(const std::string& name, std::function<void()> fn) {
    try {
        std::cout << "\n\033[96m[RUNNING " << name << "]\033[0m\n\n";
        fn();
        std::cout << "\n\033[92m[PASSED " << name << "]\033[0m\n\n";
    } catch (const std::runtime_error& ex) {
        std::cout << "\n\033[91m[FAILED " << name << "] " << ex.what() << "\033[0m\n\n";
    }
}
