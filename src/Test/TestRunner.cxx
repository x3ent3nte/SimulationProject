#include <Test/TestRunner.h>

namespace {
    constexpr char* kGreen = "\033[92m";
    constexpr char* kRed = "\033[91m";
    constexpr char* kCyan = "\033[96m";
    constexpr char* kEnd = "\033[0m";
} // namespace anonymouse

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

    std::cout << "\n" << kCyan <<"[RUNNING " << name << "]" << kEnd <<"\n\n";
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
        std::cout << "\n" << kGreen << "[PASSED " << name << "]" << kEnd << "\n\n";
    } else {
        {
            std::lock_guard<std::mutex> guard(m_mutex);
            m_failedNames.push_back(name);
        }
        std::cout << "\n" << kRed <<"[FAILED " << name << "]" << kEnd << "\n\n";
    }
}

void TestRunner::report() {
    if (m_failedNames.size() == 0) {
        std::cout << kGreen;
    } else {
        std::cout << kRed;
    }

    std::cout << "Test Report" << kEnd << "\n\n";

    std::cout << kGreen << "Passed = " << m_passedNames.size() << kEnd << "\n";
    for (const auto& name : m_passedNames) {
        std::cout << kGreen << "[PASSED " << name << "]" << kRed << "\n";
    }

    std::cout << "\n";

    std::cout << kRed << "Failed = " << m_failedNames.size() << kEnd << "\n";
    for (const auto& name : m_failedNames) {
        std::cout << kRed << "[FAILED " << name << "]" << kEnd <<"\n";
    }
}
