#include <chrono>

class Timer {
private:
    std::chrono::steady_clock::time_point begin;
public:
    Timer();

    ~Timer();
};
