#include <stdio.h>
#include <iostream>
#include <chrono>
#include "Timer.h"
#include "Reduce.cuh"
#include "Scan.cuh"

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

    constexpr int kSize = 1024 * 1024 * 128;

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

    constexpr int kSize = 1024;

    int* input = (int*) malloc(kSize * sizeof(int));
    int* output = (int*) malloc(kSize * sizeof(int));

    for (int i = 0; i < kSize; ++i) {
        input[i] = 1;
    }

    int* d_a;
    int* d_b;

    cudaMalloc(&d_a, kSize * sizeof(int));
    cudaMalloc(&d_b, kSize * sizeof(int));

    cudaMemcpy(d_a, input, kSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_b, 0, kSize * sizeof(int));

    {
        Timer timer;
        Scan::scan<int, add>(d_a, d_b,kSize);
    }

    cudaMemcpy(output, d_b, kSize * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < kSize; ++i) {
        //printf("%d %d\n", i, output[i]);
    }

    printf("\nEnd scanPlayground\n\n");
}

int main() {
    reducePlayground();
    scanPlayground();
}
