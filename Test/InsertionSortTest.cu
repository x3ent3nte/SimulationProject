#include <Test/InsertionSortTest.cuh>

#include <Kernel/InsertionSort.cuh>
#include <Timer.h>
#include <stdio.h>
#include <string>
#include <iostream>

namespace {

__device__
int intGreater(int a, int b) {
    return a > b;
}

template<typename T>
void runTest(T* nums, int size) {
    
    int* d_nums;
    int* d_needsSorting;
    cudaMalloc(&d_nums, size * sizeof(int));
    cudaMalloc(&d_needsSorting, sizeof(int));

    cudaMemcpy(d_nums, nums, size * sizeof(int), cudaMemcpyHostToDevice);

    {
        Timer timer;
        InsertionSort::sort<int, intGreater>(d_nums, d_needsSorting, size);
    }

    {
        Timer timer;
        InsertionSort::sort<int, intGreater>(d_nums, d_needsSorting, size);
    }

    {
        Timer timer;
        InsertionSort::sort<int, intGreater>(d_nums, d_needsSorting, size);
    }

    cudaMemcpy(nums, d_nums, size * sizeof(int), cudaMemcpyDeviceToHost);

    int numErrors = 0;
    for (int i = 1; i < size; ++i) {
        int left = nums[i - 1];
        int right = nums[i];

        if (left > right) {
            printf("Index %d Value %d greater than %d\n", i, left, right);
            numErrors += 1;
        }
    }

    cudaFree(d_nums);
    cudaFree(d_needsSorting);

    if (numErrors > 0) {
        throw "Num Errors: " + std::to_string(numErrors) + "\n";
    }
}

} // namespace anonymous

void InsertionSortTest::run() {
    printf("Begin InsertionSortTest\n");

    int size = 1024 * 17;
    int * nums = (int*) malloc(size * sizeof(int));

    for (int i = 0; i < size; ++i) {
        nums[i] = i % 100;
    }

    try {
        runTest<int>(nums, size);
    } catch (const std::string& ex)  {
        std::cerr << ex;
    }
    free(nums);

    printf("End InsertionSortTest\n");
}
