#ifndef SCAN_CUH
#define SCAN_CUH

#include <stdio.h>

namespace Scan {
    template<typename T, T (*FN)(T, T)>
    void scan(T* in, T* out, T* offsets, int size);
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

template<typename T, T (*FN)(T, T)>
__global__
void scanKernel(T* in, T* out, T* offsets, int size) {
    int tid = threadIdx.x;
    int globalOffset = blockDim.x * blockIdx.x;
    int gid = globalOffset + tid;

    if (gid >= size) { return; }

    extern __shared__ T sharedInts[];

    int localSize = min(blockDim.x, size - globalOffset);

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
        
        if ((prev != value) && (i != 0)) {
            printf("Outlier at %d %d prev was %d\n", i, value, prev);
        }

        prev = value;
    }

    free(offsetsHost);

    printf("Offset Size %d Total %d\n", size, total);
}

} // namespace anonymous

template<typename T, T (*FN)(T, T)>
void Scan::scan(T* in, T* out, T* offsets, int size) {

    printf("Scan size %d\n", size);
    constexpr int threadsPerBlock = 1024;
    int numBlocks = ceil(size / (float) threadsPerBlock);
    scanKernel<T, FN><<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(in, out, offsets, size);

    printOffsets(offsets, numBlocks);

    if (numBlocks > 1) {
        Scan::scan<T, FN>(offsets, offsets, offsets + numBlocks, numBlocks);
        
        int sizeOfOffsetAdd = size - threadsPerBlock;
        int numBlocksToAddOffsets = ceil(sizeOfOffsetAdd / (float) threadsPerBlock);
        
        applyBlockOffsets<T, FN><<<numBlocksToAddOffsets, threadsPerBlock>>>(out + threadsPerBlock, offsets, sizeOfOffsetAdd);
    }
}

#endif
