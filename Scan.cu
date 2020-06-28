#include "Timer.h"

#define SIZE 1024

__global__
void scan(int* in,  int* out, int size) {
    int tid = threadIdx.x;
    int globalOffset = blockDim.x * blockIdx.x;
    int gid = globalOffset + tid;

    if (gid >= size) { return; }

    extern __shared__ int sharedInts[];

    int localSize = SIZE - globalOffset;

    sharedInts[tid] = in[gid];

    __syncthreads();

    for (int offset = 1; offset < localSize; offset <<= 1) {
        int leftValue;
        int leftIndex = tid - offset;

        int validLeft = leftIndex >= 0;
        if (validLeft) {
            leftValue = sharedInts[leftIndex];
        }

        __syncthreads();

        if (validLeft) {
            sharedInts[tid] += leftValue;
        }

        __syncthreads();
    }

    out[gid] = sharedInts[tid];
}

int main() {

    int* input = (int*) malloc(SIZE * sizeof(int));
    int* output = (int*) malloc(SIZE * sizeof(int));

    for (int i = 0; i < SIZE; ++i) {
        input[i] = 1;
    }

    int* d_a;
    int* d_b;

    cudaMalloc(&d_a, SIZE * sizeof(int));
    cudaMalloc(&d_b, SIZE * sizeof(int));

    cudaMemcpy(d_a, input, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_b, 0, SIZE * sizeof(int));

    {
        Timer timer;
        scan<<<1, SIZE, SIZE * sizeof(int)>>>(d_a, d_b, SIZE);
    }

    cudaMemcpy(output, d_b, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; ++i) {
        printf("%d %d\n", i, output[i]);
    }
}
