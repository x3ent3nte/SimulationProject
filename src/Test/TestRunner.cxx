#include <Test/TestRunner.h>

void TestInstance::fail() {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_passed = false;
}

bool TestInstance::hasPassed() {
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_passed;
}

void TestInstance::assertTrue(bool b) {
    if (!b) {
        fail();
        throw std::runtime_error("Expected True");
    }
}

void TestRunner::test(
    const std::string& name,
    std::function<void(std::shared_ptr<TestInstance> testInstance)> fn) {

    std::cout << "\n\033[96m[RUNNING " << name << "]\033[0m\n\n";
    auto testInstance = std::make_shared<TestInstance>();

    try {
        fn(testInstance);
    } catch (const std::runtime_error& ex) {
        std::cout << "Test exception = " << ex.what() << "\n";
    }

    if (testInstance->hasPassed()) {
        {
            std::lock_guard<std::mutex> guard(m_mutex);
            m_passedNames.push_back(name);
        }
        std::cout << "\n\033[92m[PASSED " << name << "]\033[0m\n\n";
    } else {
        {
            std::lock_guard<std::mutex> guard(m_mutex);
            m_failedNames.push_back(name);
        }
        std::cout << "\n\033[91m[FAILED " << name << "]\033[0m\n\n";
    }
}

void TestRunner::report() {
    if (m_numberFailed == 0) {
        std::cout << "\033[92m";
    } else {
        std::cout << "\033[91m";
    }

    std::cout << "Test Report\n\n";

    std::cout << "Passed = " << m_passedNames.size() << "\n";
    for (const auto& name : m_passedNames) {
        std::cout << "\n\033[92m[" << name << "]\033[0m\n\n";
    }

    std::cout << "Failed = " << m_failedNames.size() << "\n";
    for (const auto& name : m_failedNames) {
        std::cout << "\n\033[91m[" << name << "]\033[0m\n\n";
    }

    std::cout << "\033[0m";
}
