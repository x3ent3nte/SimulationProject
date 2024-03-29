#include <Utils/Timer.h>
#include <Renderer/Application.h>
#include <Cuda/Reduce.cuh>
#include <Cuda/Scan.cuh>
#include <Cuda/RadixSort.cuh>
#include <Cuda/Agent.cuh>
#include <Cuda/ContinuousCollision.cuh>
#include <Cuda/CudaSimulator.cuh>
#include <Test/InsertionSortCudaTest.cuh>

#include <stdio.h>
#include <iostream>
#include <memory>

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
        Timer timer("Reduce GPU");
        int result = Reduce::reduce<int, add>(d_in, d_out, kSize);
        printf("\nGPU result: %d\n", result);
    }

    {
        Timer timer("Reduce SER");
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

    int* d_data;

    checkCudaErrors(cudaMalloc(&d_data, kSize * sizeof(int) * 2));

    cudaMemcpy(d_data, input, kSize * sizeof(int), cudaMemcpyHostToDevice);

    {
        Timer timer("GPU Scan");
        for (int i = 0; i < 10; ++i) {
            Scan::scan<int, add>(d_data, kSize);
            checkScanErrors(input, output, d_data, kSize);
        }
    }

    free(input);
    free(output);
    cudaFree(d_data);

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
    uint4* d_flags;

    cudaMalloc(&d_a, kSize * sizeof(unsigned int));
    cudaMalloc(&d_b, kSize * sizeof(unsigned int));
    cudaMalloc(&d_flags, kSize * sizeof(uint4) * 2);

    cudaMemcpy(d_a, input, kSize * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(d_b, 0, kSize * sizeof(unsigned int));

    unsigned int* sorted;
    {
        Timer timer("GPU Radix Sort");
        sorted = RadixSort::sort<unsigned int>(d_a, d_b, d_flags, kSize);
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
    cudaFree(d_flags);

    printf("\nEnd radixSortPlayground\n");
}

void extractResultsFromCudaSimulator(CudaAgent* agents, float3* positions, size_t size) {
    float3* h_positions = (float3*) malloc(size * sizeof(float3));

    cudaMemcpy(h_positions, positions, size * sizeof(float3), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < size; ++i) {
        //float3 v = h_positions[i];
        //std::cout << "i " << i << " " << v.x << " " << v.y << " " << v.z << "\n";
    }

    free(h_positions);
}

void cudaSimulator() {
    printf("\nBegin Cuda Simulator\n");

    size_t xDim = 512;
    size_t numElements = 32 * xDim;

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
        Timer timer("Cuda Simulator");

        CudaSimulator::simulate(d_agents, d_positions, numElements);
        extractResultsFromCudaSimulator(d_agents, d_positions, numElements);
    }

    free(agents);
    cudaFree(d_agents);
    cudaFree(d_positions);

    printf("\nEnd Cuda Simulator\n");
}

// TODO
// For some mysterious reason, reduce and scan are non deterministic and suffer from errors when threadsPerBlock is not 1024

int main() {
    srand(time(NULL));

    return Application::create()->run();

    //cudaSimulator();
    //reducePlayground();
    //scanPlayground();
    //radixSortPlayground();
}
