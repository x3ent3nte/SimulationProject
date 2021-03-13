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

    int* d_in;
    int* d_out;
    int* d_offsets;

    cudaMalloc(&d_in, bufferSize);
    cudaMalloc(&d_out, bufferSize);
    cudaMalloc(&d_offsets, bufferSize);

    cudaMemcpy(d_in, data.data(), bufferSize, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, bufferSize);
    cudaMemset(d_offsets, 0, bufferSize);

    {
        Timer timer("Scan CUDA");
        Scan::scan<int, add>(d_in, d_out, d_offsets, data.size());
    }

    std::vector<int> result(data.size());
    cudaMemcpy(result.data(), d_out, bufferSize, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_offsets);

    return result;
}
