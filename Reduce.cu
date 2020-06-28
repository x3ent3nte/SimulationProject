#include <stdio.h>
#include <iostream>
#include <chrono>
#include "Timer.h"

#define SIZE 1024 * 1024 * 4
#define THREADS_PER_BLOCK 256

int serialReduce(int* ints, int size) {
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += ints[i];
    }
    return sum;
}

__host__
__device__
int add(int a, int b) {
    return a + b;
}

template<typename T, T (*FN)(T, T)>
__global__
void reduce(T* in, T* out, int size, int (*fn)(int, int)) {
    extern __shared__ int sharedInts[];

    int tid = threadIdx.x;
    int blockStart = blockDim.x * blockIdx.x * 2;
    int gid = blockStart + tid;

    if (gid >= size) { return; }

    sharedInts[tid] = in[gid];

    int firstReduceIndex = gid + blockDim.x;
    if (firstReduceIndex < size) {
        sharedInts[tid] = FN(sharedInts[tid], in[firstReduceIndex]);
    }

    __syncthreads();

    int localSize = min(blockDim.x, size - blockStart);
    for (int offset = localSize / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sharedInts[tid] = FN(sharedInts[tid], sharedInts[tid + offset]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sharedInts[0];
    }
}

template <typename T, T (*FN)(T, T)>
int runReduce(int* d_a, int* d_b, int size) {
    while (size > 1) {
        int numBlocks = ceil(size / ((float) THREADS_PER_BLOCK * 2));
        reduce<T, FN><<<numBlocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(d_a, d_b, size, add);
        size = numBlocks;

        printf("Size is %d\n", size);

        int* temp = d_a;
        d_a = d_b;
        d_b = temp;
    }

    int result = 0;
    cudaMemcpy(&result, d_a, sizeof(int), cudaMemcpyDeviceToHost);
    return result;
}

int main() {

    int* in = (int*) malloc(SIZE *sizeof(int));

    for (int i = 0; i < SIZE; ++i) {
        in[i] = 1;
    }

    int* d_in;
    int* d_out;

    cudaMalloc(&d_in, SIZE * sizeof(int));
    cudaMalloc(&d_out, SIZE * sizeof(int));

    cudaMemcpy(d_in, in, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, SIZE * sizeof(int));

    {
        Timer time;
        int result = runReduce<int, add>(d_in, d_out, SIZE);
        printf("GPU result: %d\n", result);
    }

    {
        Timer time;
        int result = serialReduce(in, SIZE);
        printf("SER result: %d\n", result);
    }

    free(in);

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
