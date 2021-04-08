#ifndef APPLICATION_H
#define APPLICATION_H

#include <memory>

class Application {
public:

    virtual ~Application() = default;

    virtual int run() = 0;

    static std::shared_ptr<Application> create();
};

#endif
