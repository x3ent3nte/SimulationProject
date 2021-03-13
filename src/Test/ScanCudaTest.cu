#include <Test/ScanCudaTest.cuh>

#include <Cuda/Scan.cuh>
#include <Utils/Timer.h>

namespace {
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

    {
        Timer timer("Scan CUDA");
        Scan::scan<int, add>(d_data, data.size());
    }

    std::vector<int> result(data.size());
    cudaMemcpy(result.data(), d_data, bufferSize, cudaMemcpyDeviceToHost);

    cudaFree(d_data);

    return result;
}
