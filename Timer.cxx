#include "Timer.h"

#include <iostream>
#include <chrono>

Timer::Timer()
: begin(std::chrono::steady_clock::now()) {};

Timer::~Timer() {
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Duration: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " microseconds\n";
}
