#ifndef REDUCE_TEST_H
#define REDUCE_TEST_H

#include <Test/ReduceVulkanTest.h>
#include <Test/ReduceCudaTest.cuh>

class ReduceTest {
private:

public:

    virtual ~ReduceTest();

    void run();
};

#endif
