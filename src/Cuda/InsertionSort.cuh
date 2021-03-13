#ifndef INSERTION_SORT_CUH
#define INSERTION_SORT_CUH

#include <Utils/Timer.h>

#include <iostream>

namespace InsertionSort {
    template<typename T, int (*FN)(T, T)>
    void sort(T* elements, int* needsSortingFlag, int size);
}

namespace {

template<typename T, int (*FN)(T, T)>
__device__
void compareHelper(T* elements, int* needsSortingFlag, int* wasSwappedFlag, int leftIndex, int rightIndex) {
    T left = elements[leftIndex];
    T right = elements[rightIndex];

    if (FN(left, right)) {
        elements[leftIndex] = right;
        elements[rightIndex] = left;
        needsSortingFlag[0] = 1;
        wasSwappedFlag[0] = 1;
    }
}

template<typename T, int (*FN)(T, T)>
__device__
void insertionSortHelper(T* elements, int* wasSwappedFlag, int leftIndex, int size) {

    int rightIndex = leftIndex + 1;
    int rightIndexPlusOne = rightIndex + 1;

    __shared__ int needsSortingFlag[1];

    do {
        __syncthreads();

        needsSortingFlag[0] = 0;

        __syncthreads();

        if (rightIndex < size) {
            compareHelper<T, FN>(elements, needsSortingFlag, wasSwappedFlag, leftIndex, rightIndex);
        }

        __syncthreads();

        if (rightIndexPlusOne < size) {
            compareHelper<T, FN>(elements, needsSortingFlag, wasSwappedFlag, rightIndex, rightIndexPlusOne);
        }

        __syncthreads();
    } while (needsSortingFlag[0]);
}

template<typename T, int (*FN)(T, T), int THREADS_PER_BLOCK>
__global__
void insertionSortKernel(T* elements, int* wasSwappedFlag, int size) {

    __shared__ T sharedElements[2 * THREADS_PER_BLOCK * sizeof(T)];

    int globalOffset = blockDim.x * blockIdx.x * 2;
    int leftIndex = threadIdx.x * 2;
    int globalLeftIndex = globalOffset + leftIndex;

    if (globalLeftIndex >= size) { return; }

    int localSize = min(2 * THREADS_PER_BLOCK, size - globalOffset);

    sharedElements[leftIndex] = elements[globalLeftIndex];

    int globalRightIndex = globalLeftIndex + 1;
    int rightIndex = leftIndex + 1;
    if (rightIndex < localSize) {
        sharedElements[rightIndex] = elements[globalRightIndex];
    }

    insertionSortHelper<T, FN>(sharedElements, wasSwappedFlag, leftIndex, localSize);

    elements[globalLeftIndex] = sharedElements[leftIndex];
    if (rightIndex < localSize) {
        elements[globalRightIndex] = sharedElements[rightIndex];
    }
}

int needsSorting(int* needsSortingFlag) {
    int needsSortingFlagCPU;
    cudaMemcpy(&needsSortingFlagCPU, needsSortingFlag, sizeof(int), cudaMemcpyDeviceToHost);
    return needsSortingFlagCPU;
}

} // namespace anonymous

template<typename T, int (*FN)(T, T)>
void InsertionSort::sort(T* elements, int* needsSortingFlag, int size) {

    constexpr int threadsPerBlock = 256;
    int offset = threadsPerBlock;

    int numBlocks = ceil(size / (float) (threadsPerBlock * 2));

    int numIterations = 0;

    {
        Timer timer("Insertion Sort CUDA");
        do {
            cudaMemset(needsSortingFlag, 0, sizeof(int));

            insertionSortKernel<T, FN, threadsPerBlock><<<numBlocks, threadsPerBlock>>>(elements, needsSortingFlag, size);
            insertionSortKernel<T, FN, threadsPerBlock><<<numBlocks, threadsPerBlock>>>(elements + offset, needsSortingFlag, size - offset);

            numIterations += 1;
        } while (needsSorting(needsSortingFlag));
    }
    //std::cout << "Insertion Sort Cuda total number of iterations = " << numIterations << "\n";
}

#endif
