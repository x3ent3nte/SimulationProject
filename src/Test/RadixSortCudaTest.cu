#include <Test/RadixSortCudaTest.cuh>

#include <Cuda/RadixSort.cuh>

#include <Utils/Timer.h>

std::vector<uint32_t> RadixSortCudaTest::run(const std::vector<uint32_t>& numbers) {

    unsigned int* d_a;
    unsigned int* d_b;
    uint4* d_flags;

    const size_t memorySize = numbers.size() * sizeof(uint32_t);

    cudaMalloc(&d_a, memorySize);
    cudaMalloc(&d_b, memorySize);
    cudaMalloc(&d_flags, numbers.size() * sizeof(uint4) * 2);

    cudaMemcpy(d_a, numbers.data(), memorySize, cudaMemcpyHostToDevice);
    cudaMemset(d_b, 0, memorySize);
    cudaMemset(d_flags, 0, numbers.size() * sizeof(uint4) * 2);

    uint32_t* d_sorted;
    {
        Timer timer("Radix Sort Cuda");
        d_sorted = RadixSort::sort<unsigned int>(d_a, d_b, d_flags, numbers.size());
    }

    std::vector<uint32_t> sorted(numbers.size());
    cudaMemcpy(sorted.data(), d_sorted, memorySize, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_flags);

    return sorted;
}
