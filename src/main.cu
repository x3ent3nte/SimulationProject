#include <stdio.h>
#include <iostream>

#include <Timer.h>
#include <Renderer/Renderer.h>
#include <Kernel/Reduce.cuh>
#include <Kernel/Scan.cuh>
#include <Kernel/RadixSort.cuh>
#include <Kernel/Agent.cuh>
#include <Kernel/ContinuousCollision.cuh>
#include <Kernel/CudaSimulator.cuh>
#include <Test/InsertionSortTest.cuh>

#define checkCudaErrors(call)                                   \
do {                                                            \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
        cudaGetErrorString(err));                               \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while (0)

__device__
int add(int a, int b) {
    return a + b;
}

int serialReduce(int* ints, int size) {
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += ints[i];
    }
    return sum;
}

void reducePlayground() {
    printf("\nBegin reducePlayground\n");

    constexpr int kSize = 1024 * 1024 * 32;

    int* in = (int*) malloc(kSize *sizeof(int));

    for (int i = 0; i < kSize; ++i) {
        in[i] = 1;
    }

    int* d_in;
    int* d_out;

    cudaMalloc(&d_in, kSize * sizeof(int));
    cudaMalloc(&d_out, kSize * sizeof(int));

    cudaMemcpy(d_in, in, kSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, kSize * sizeof(int));

    {
        Timer time("Reduce GPU");
        int result = Reduce::reduce<int, add>(d_in, d_out, kSize);
        printf("\nGPU result: %d\n", result);
    }

    {
        Timer time("Reduce SER");
        int result = serialReduce(in, kSize);
        printf("\nSER result: %d\n", result);
    }

    free(in);

    cudaFree(d_in);
    cudaFree(d_out);

    printf("\nEnd reducePlayground\n\n");
}

void checkScanErrors(int* input, int * output, int* d_out, int size) {
    cudaMemcpy(output, d_out, size * sizeof(int), cudaMemcpyDeviceToHost);

    int expected = 0;
    int numErrors = 0;
    for (int i = 0; i < size; ++i) {
        expected += input[i];

        int actual = output[i];
        if (expected != actual) {
            //printf("Mismatch i %d exp %d act %d\n", i, expected, actual);
            numErrors += 1;
        }
    }

    if (numErrors > 0) {
        printf("Num scan errors %d\n", numErrors);
    }
}

void scanPlayground() {
    printf("\nBegin scanPlayground\n");

    constexpr int kSize = 1024 * 1024 * 4;

    int* input = (int*) malloc(kSize * sizeof(int));
    int* output = (int*) malloc(kSize * sizeof(int));

    for (int i = 0; i < kSize; ++i) {
        input[i] = 1;
    }

    int* d_in;
    int* d_out;
    int* d_offsets;

    checkCudaErrors(cudaMalloc(&d_in, kSize * sizeof(int)));
    cudaMalloc(&d_out, kSize * sizeof(int));
    cudaMalloc(&d_offsets, kSize * sizeof(int));

    cudaMemcpy(d_in, input, kSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, kSize * sizeof(int));
    cudaMemset(d_offsets, 0, kSize * sizeof(int));

    {
        Timer timer("GPU Scan");
        for (int i = 0; i < 10; ++i) {
            Scan::scan<int, add>(d_in, d_out, d_offsets, kSize);
            checkScanErrors(input, output, d_out, kSize);
        }
    }

    free(input);
    free(output);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_offsets);

    printf("\nEnd scanPlayground\n\n");
}

void radixSortPlayground() {

    printf("\nBegin radixSortPlayground\n");

    constexpr int kSize = 1024 * 32;

    unsigned int* input = (unsigned int*) malloc(kSize * sizeof(unsigned int));
    unsigned int* output = (unsigned int*) malloc(kSize * sizeof(unsigned int));

    for (int i = 0; i < kSize; ++i) {
        input[i] = 100 - (i % 100);
        output[i] = 0;
    }

    unsigned int* d_a;
    unsigned int* d_b;
    uint4* d_flags_a;
    uint4* d_flags_b;

    cudaMalloc(&d_a, kSize * sizeof(unsigned int));
    cudaMalloc(&d_b, kSize * sizeof(unsigned int));
    cudaMalloc(&d_flags_a, kSize * sizeof(uint4));
    cudaMalloc(&d_flags_b, kSize * sizeof(uint4));

    cudaMemcpy(d_a, input, kSize * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(d_b, 0, kSize * sizeof(unsigned int));

    unsigned int* sorted;
    {
        Timer timer("GPU Radix Sort");
        sorted = RadixSort::sort<unsigned int>(d_a, d_b, d_flags_a, d_flags_b, kSize);
    }

    cudaMemcpy(output, sorted, kSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    int numErrors = 0;
    for (int i = 1; i < kSize; ++i) {
        unsigned int left = output[i - 1];
        unsigned int right = output[i];

        if (left > right) {
            printf("Radix Sort Mismatch at Index %d Left %d Right %d\n", i, left, right);
            numErrors += 1;
        }
    }

    printf("Radix Sort numErrors %d\n", numErrors);

    free(input);
    free(output);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_flags_a);
    cudaFree(d_flags_b);

    printf("\nEnd radixSortPlayground\n");
}

void cudaSimulator() {
    printf("\nBegin Cuda Simulator\n");

    size_t xDim = 512;
    size_t numElements = 128 * xDim;

    CudaAgent* agents = (CudaAgent*) malloc(numElements * sizeof(CudaAgent));
    for (size_t i = 0; i < numElements; ++i) {
        float fi = (float) i;
        agents[i] = CudaAgent{float3{fi, fi, fi,}, float3{fi + 100, fi + 100, fi + 100}};
    }

    CudaAgent* d_agents;
    cudaMalloc(&d_agents, numElements * sizeof(CudaAgent));
    cudaMemcpy(d_agents, agents, numElements * sizeof(CudaAgent), cudaMemcpyHostToDevice);

    float3* d_positions;
    cudaMalloc(&d_positions, numElements * sizeof(float3));

    {
        Timer time("Cuda Simulator");

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        CudaSimulator::simulate(d_agents, d_positions, numElements);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "xxx duration: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " microseconds\n";
    }

    free(agents);
    cudaFree(d_agents);
    cudaFree(d_positions);

    printf("\nEnd Cuda Simulator\n");
}

// TODO
// For some mysterious reason, reduce and scan are non deterministic and suffer from errors when threadsPerBlock is not 1024

int main() {
    Renderer().render();
    cudaSimulator();
    //reducePlayground();
    //scanPlayground();
    //InsertionSortTest::run();
    //radixSortPlayground();
}
