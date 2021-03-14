#include <Test/ScanCudaTest.cuh>

#include <Cuda/Scan.cuh>
#include <Utils/Timer.h>
#include <iostream>

namespace {
    __host__
    __device__
    int add(int a, int b) {
        return a + b;
    }
} // namespace anonymous

std::vector<int> ScanCudaTest::run(const std::vector<int>& data) {
    size_t bufferSize = data.size() * sizeof(int);

    int* d_data;

    cudaMalloc(&d_data, bufferSize * 2);
    cudaMemcpy(d_data, data.data(), bufferSize, cudaMemcpyHostToDevice);

    int x;
    {
        Timer timer("Scan CUDA");
        x = Scan::scan<int, add>(d_data, data.size());
    }

    std::cout << "X is " << x << "\n";

    std::vector<int> result(data.size());
    cudaMemcpy(result.data(), d_data, bufferSize, cudaMemcpyDeviceToHost);

    cudaFree(d_data);

    return result;
}
