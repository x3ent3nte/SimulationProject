#include <Test/TestInstance.h>

void TestInstance::assertTrue(bool b) {
    if (!b) {
        throw std::runtime_error("Expected True");
    }
}

void TestInstance::test(
    const std::string& name,
    std::function<void()> fn) {

    try {
        std::cout << "\n\033[96m[RUNNING " << name << "]\033[0m\n\n";
        fn();
        std::cout << "\n\033[92m[PASSED " << name << "]\033[0m\n\n";
        m_numberPassed += 1;
    } catch (const std::runtime_error& ex) {
        std::cout << "\n\033[91m[FAILED " << name << "] " << ex.what() << "\033[0m\n\n";
        m_numberFailed += 1;
    }
}

void TestInstance::printReport() {

    if (m_numberFailed == 0) {
        std::cout << "\033[92m";
    } else {
        std::cout << "\033[91m";
    }

    std::cout << "Test Report\n\n";

    std::cout << "Passed = " << m_numberPassed << "\n";
    std::cout << "Failed = " << m_numberFailed << "\n";

    std::cout << "\033[0m";
}
