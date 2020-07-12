#include <stdio.h>
#include <iostream>
#include "Timer.h"
#include "Kernel/Reduce.cuh"
#include "Kernel/Scan.cuh"
#include "Kernel/RadixSort.cuh"
#include "Test/InsertionSortTest.cuh"

#define checkCudaErrors(call)                                \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
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
    printf("Begin reducePlayground\n");

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
        Timer time;
        int result = Reduce::reduce<int, add>(d_in, d_out, kSize);
        printf("\nGPU result: %d\n", result);
    }

    {
        Timer time;
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
    printf("Begin scanPlayground\n");

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
        Timer timer;
        for (int i = 0; i < 100; ++i) {
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

    printf("Begin radixSortPlayground\n");

    constexpr int kSize = 1024 * 1024;
    
    unsigned int* input = (unsigned int*) malloc(kSize * sizeof(unsigned int));
    unsigned int* output = (unsigned int*) malloc(kSize * sizeof(unsigned int));

    for (int i = 0; i < kSize; ++i) {
        input[i] = i % 100;
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
        Timer timer;
        sorted = RadixSort::sort<unsigned int>(d_a, d_b, d_flags_a, d_flags_b, kSize);
    }

    cudaMemcpy(output, sorted, kSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    free(input);
    free(output);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_flags_a);
    cudaFree(d_flags_b);

    printf("\nEnd radixSortPlayground\n");
}

// For some mysterious reason, reduce and scan are non deterministic and suffer from errors when threadsPerBlock is not 1024

int main() {
    reducePlayground();
    scanPlayground();
    InsertionSortTest::run();
    //radixSortPlayground();
}
