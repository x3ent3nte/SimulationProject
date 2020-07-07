#include <stdio.h>
#include <iostream>
#include <chrono>
#include "Timer.h"
#include "Reduce.cuh"
#include "Scan.cuh"
#include "InsertionSort.cuh"

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

void scanPlayground() {
    printf("Begin scanPlayground\n");

    constexpr int kSize = 1021 * 1079 * 3;

    int* input = (int*) malloc(kSize * sizeof(int));
    int* output = (int*) malloc(kSize * sizeof(int));

    for (int i = 0; i < kSize; ++i) {
        input[i] = 1;
    }

    int* d_in;
    int* d_out;
    int* d_offsets;

    cudaMalloc(&d_in, kSize * sizeof(int));
    cudaMalloc(&d_out, kSize * sizeof(int));
    cudaMalloc(&d_offsets, kSize * sizeof(int));

    cudaMemcpy(d_in, input, kSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, kSize * sizeof(int));
    cudaMemset(d_offsets, 0, kSize * sizeof(int));

    {
        Timer timer;
        Scan::scan<int, add>(d_in, d_out, d_offsets, kSize);
    }

    cudaMemcpy(output, d_out, kSize * sizeof(int), cudaMemcpyDeviceToHost);

    int expected = 0;
    int numErrors = 0;
    for (int i = 0; i < kSize; ++i) {
        expected += input[i];

        int actual = output[i];
        if (expected != actual) {
            //printf("Mismatch i %d exp %d act %d\n", i, expected, actual);
            numErrors += 1;
        }
    }

    printf("Num scan errors %d\n", numErrors);

    free(input);
    free(output);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_offsets);

    printf("\nEnd scanPlayground\n\n");
}

__device__
int intGreater(int a, int b) {
    return a > b;
}

void insertionSortPlayground() {
    printf("Begin insertionSortPlayground\n");

    constexpr int kSize = 1024 * 17;
    int * nums = (int*) malloc(kSize * sizeof(int));

    for (int i = 0; i < kSize; ++i) {
        nums[i] = i % 100;
    }

    int* d_nums;
    int* d_needsSorting;
    cudaMalloc(&d_nums, kSize * sizeof(int));
    cudaMalloc(&d_needsSorting, sizeof(int));

    cudaMemcpy(d_nums, nums, kSize * sizeof(int), cudaMemcpyHostToDevice);

    {
        Timer timer;
        InsertionSort::sort<int, intGreater>(d_nums, d_needsSorting, kSize);
    }

    {
        Timer timer;
        InsertionSort::sort<int, intGreater>(d_nums, d_needsSorting, kSize);
    }

    {
        Timer timer;
        InsertionSort::sort<int, intGreater>(d_nums, d_needsSorting, kSize);
    }

    cudaMemcpy(nums, d_nums, kSize * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 1; i < kSize; ++i) {
        int left = nums[i - 1];
        int right = nums[i];

        if (left > right) {
            printf("Index %d Value %d greater than %d\n", i, left, right);
        }
    }

    free(nums);

    cudaFree(d_nums);
    cudaFree(d_needsSorting);

    printf("\nEnd insertionSortPlayground\n");
}

// For some mysterious reason, reduce and scan are non deterministic and suffer from errors when threadsPerBlock is not 1024

int main() {
    reducePlayground();
    scanPlayground();
    insertionSortPlayground();
}
