#include <Test/InsertionSortCudaTest.cuh>

#include <Kernel/InsertionSort.cuh>
#include <Utils/Timer.h>

#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>

namespace {

struct ValueAndIndex {
    float value;
    uint32_t index;
};

__device__
int valueAndIndexShouldSwap(ValueAndIndex a, ValueAndIndex b) {
    return a.value > b.value;
}

} // namespace anonymous

std::vector<float> InsertionSortCudaTest::run(const std::vector<float>& data) {

    std::vector<ValueAndIndex> valueAndIndexes(data.size());
    for (uint32_t i = 0; i < data.size(); ++i) {
        valueAndIndexes[i] = {data[i], i};
    }

    size_t bufferSize = valueAndIndexes.size() * sizeof(ValueAndIndex);

    ValueAndIndex* d_valueAndIndexes;
    int* d_needsSorting;
    cudaMalloc(&d_valueAndIndexes, bufferSize);
    cudaMalloc(&d_needsSorting, sizeof(int));

    cudaMemcpy(d_valueAndIndexes, valueAndIndexes.data(), bufferSize, cudaMemcpyHostToDevice);

    InsertionSort::sort<ValueAndIndex, valueAndIndexShouldSwap>(d_valueAndIndexes, d_needsSorting, valueAndIndexes.size());

    cudaMemcpy(valueAndIndexes.data(), d_valueAndIndexes, bufferSize, cudaMemcpyDeviceToHost);

    cudaFree(d_valueAndIndexes);
    cudaFree(d_needsSorting);

    std::vector<float> sorted(valueAndIndexes.size());
    for (int i = 0; i < valueAndIndexes.size(); ++i) {
        sorted[i] = (valueAndIndexes[i].value);
    }

    return sorted;
}
