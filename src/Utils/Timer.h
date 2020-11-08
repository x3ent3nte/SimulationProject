#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <string>

class Timer {
private:

    std::string m_label;
    std::chrono::steady_clock::time_point m_begin;

public:

    Timer(const std::string& label);

    ~Timer();
};

#endif
