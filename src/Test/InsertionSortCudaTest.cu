#include <Test/InsertionSortCudaTest.cuh>

#include <Kernel/InsertionSort.cuh>
#include <Utils/Timer.h>

#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>

namespace {

__device__
int floatGreater(float a, float b) {
    return a > b;
}

} // namespace anonymous

std::vector<float> InsertionSortCudaTest::run(const std::vector<float>& data) {

    std::vector<float> dataCopy(data);

    size_t size = dataCopy.size() * sizeof(float);

    float* d_data;
    int* d_needsSorting;
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_needsSorting, sizeof(int));

    cudaMemcpy(d_data, dataCopy.data(), size, cudaMemcpyHostToDevice);

    InsertionSort::sort<float, floatGreater>(d_data, d_needsSorting, size);

    cudaMemcpy(dataCopy.data(), d_data, size, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_needsSorting);

    return dataCopy;
}
