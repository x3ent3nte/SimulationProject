#include <Timer.h>

#include <iostream>
#include <chrono>

Timer::Timer(const std::string& label)
: m_label(label)
, m_begin(std::chrono::steady_clock::now()) {};

Timer::~Timer() {
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << m_label << " duration: " << std::chrono::duration_cast<std::chrono::microseconds>(end - m_begin).count() << " microseconds\n";
}
