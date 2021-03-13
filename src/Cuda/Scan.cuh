#ifndef SCAN_CUH
#define SCAN_CUH

#include <stdio.h>

namespace Scan {
    template<typename T, T (*FN)(T, T)>
    void scan(T* data, int size);
}

namespace {

template<typename T, T (*FN)(T, T)>
__global__
void applyBlockOffsets(T* ints, T* offsets, int size) {
    int tid = threadIdx.x;
    int globalOffset = blockDim.x * blockIdx.x;
    int gid = globalOffset + tid;

    if (gid >= size) { return; }

    ints[gid] = FN(ints[gid], offsets[blockIdx.x]);
}

template<typename T, T (*FN)(T, T), int THREADS_PER_BLOCK>
__global__
void scanKernel(T* in, T* out, T* offsets, int size) {
    int tid = threadIdx.x;
    int globalOffset = blockDim.x * blockIdx.x;
    int gid = globalOffset + tid;

    if (gid >= size) { return; }

    __shared__ T sharedInts[THREADS_PER_BLOCK * sizeof(T)];

    int localSize = min(blockDim.x, size - globalOffset);

    sharedInts[tid] = in[gid];

    __syncthreads();

    for (int offset = 1; offset < localSize; offset <<= 1) {
        T leftValue;
        int leftIndex = tid - offset;

        int validLeft = leftIndex >= 0;
        if (validLeft) {
            leftValue = sharedInts[leftIndex];
        }

        __syncthreads();

        if (validLeft) {
            sharedInts[tid] = FN(sharedInts[tid], leftValue);
        }

        __syncthreads();
    }

    out[gid] = sharedInts[tid];

    if ((tid + 1) == localSize) {
        offsets[blockIdx.x] = sharedInts[tid];
    }
}

void printOffsets(int* offsets, int size) {
    int* offsetsHost = (int*) malloc(size * sizeof(int));
    cudaMemcpy(offsetsHost, offsets, size * sizeof(int), cudaMemcpyDeviceToHost);

    int prev = -1;
    int total = 0;
    for (int i = 0; i < size; ++i) {
        int value = offsetsHost[i];
        total += value;

        if ((prev != value)) {
            printf("XXXX Outlier at %d %d prev was %d\n", i, value, prev);
        }

        prev = value;
    }

    free(offsetsHost);

    printf("Offset Size %d Total %d\n", size, total);
}

} // namespace anonymous

template<typename T, T (*FN)(T, T)>
void Scan::scan(T* data, int size) {

    T* offsets = data + size;
    //printf("Scan size %d\n", size);
    constexpr int threadsPerBlock = 32;
    int numBlocks = ceil(size / (float) threadsPerBlock);
    scanKernel<T, FN, threadsPerBlock><<<numBlocks, threadsPerBlock>>>(data, data, offsets, size);

    //printOffsets(offsets, numBlocks);

    if (numBlocks > 1) {
        Scan::scan<T, FN>(offsets, numBlocks);

        applyBlockOffsets<T, FN><<<numBlocks - 1, threadsPerBlock>>>(data + threadsPerBlock, offsets, size - threadsPerBlock);
    }
}

#endif
