#include <Test/TestRunner.h>

#include <Utils/TextColour.h>

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

    std::cout << "\n" << TextColour::CYAN << "[RUNNING " << name << "]" << TextColour::END << "\n\n";
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
        std::cout << "\n" << TextColour::GREEN << "[PASSED " << name << "]" << TextColour::END << "\n\n";
    } else {
        {
            std::lock_guard<std::mutex> guard(m_mutex);
            m_failedNames.push_back(name);
        }
        std::cout << "\n" << TextColour::RED << "[FAILED " << name << "]" << TextColour::END << "\n\n";
    }
}

void TestRunner::report() {
    if (m_failedNames.size() == 0) {
        std::cout << TextColour::GREEN;
    } else {
        std::cout << TextColour::RED;
    }

    std::cout << "\nTest Report" << TextColour::END << "\n\n";

    std::cout << TextColour::GREEN << "Passed = " << m_passedNames.size() << TextColour::END << "\n";
    for (const auto& name : m_passedNames) {
        std::cout << TextColour::GREEN << "[PASSED " << name << "]" << TextColour::RED << "\n";
    }

    std::cout << "\n";

    if (m_failedNames.size() > 0) {
        std::cout << TextColour::RED << "Failed = " << m_failedNames.size() << TextColour::END << "\n";
        for (const auto& name : m_failedNames) {
            std::cout << TextColour::RED << "[FAILED " << name << "]" << TextColour::END << "\n";
        }
    }
}
