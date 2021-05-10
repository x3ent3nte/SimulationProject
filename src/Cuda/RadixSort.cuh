#ifndef RADIX_SORT_CUH
#define RADIX_SORT_CUH

#include <Cuda/Scan.cuh>
#include <set>

namespace RadixSort {
    template<typename T>
    T* sort(T* a, T* b, uint4* flags, int size);
}

namespace {

__device__
uint4 uint4Add(uint4 a, uint4 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

template<typename T>
__global__
void mark(const T* elements, uint4* flags, int pos, int size) {

    int tid = threadIdx.x;
    int globalOffset = blockDim.x * blockIdx.x;
    int gid = globalOffset + tid;

    if (gid >= size) { return; }

    T index = (elements[gid] >> pos) & 3;

    uint4 flag = {0, 0, 0, 0};
    *((&flag.x) + index) = 1;

    flags[gid] = flag;
}

template<typename T>
__global__
void scatter(const T* in, T* out, const uint4* addresses, uint4 totalOffset, int pos, int size) {
    int tid = threadIdx.x;
    int globalOffset = blockDim.x * blockIdx.x;
    int gid = globalOffset + tid;

    if (gid >= size) { return; }

    T value = in[gid];

    T index = (value >> pos) & 3;

    uint4 addressVector;
    if (gid == 0) {
        addressVector = {0, 0, 0, 0};
    } else {
        addressVector = addresses[gid - 1];
    }

    unsigned int address = *((&addressVector.x) + index);
    unsigned int offset = *((&totalOffset.x) + index);
    address += offset;

    out[address] = value;
}

void printFlags(uint4* flags, int size) {
    uint4* flagsHost = (uint4*) malloc(sizeof(uint4) * size);
    cudaMemcpy(flagsHost, flags, sizeof(uint4) * size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        uint4 flag = flagsHost[i];
        printf("Flags at index %i: %d %d %d %d\n", i, flag.x, flag.y, flag.z, flag.w);
    }
}

void printAddress(unsigned int* addresses, int size) {
    unsigned int* addressesHost = (unsigned int*) malloc(sizeof(unsigned int) * size);
    cudaMemcpy(addressesHost, addresses, sizeof(unsigned int) * size, cudaMemcpyDeviceToHost);

    std::set<unsigned int> usedAddresses;

    for (int i = 0; i < size; ++i) {
        unsigned int address = addressesHost[i];
        printf("Address: %d %d\n", i, address);

        if (usedAddresses.count(address)) {
            printf("Duplicate at %d address %d\n", i, address);
        } else {
            usedAddresses.insert(address);
        }
    }
}

} // namespace anonymous

template<typename T>
T* RadixSort::sort(T* a, T* b, uint4* flags, int size) {

    constexpr int numBits = sizeof(T) * 8;
    constexpr int threadsPerBlock = 1024;
    const int numBlocks = ceil(size / (float) threadsPerBlock);

    for (int pos = 0; pos < numBits; pos += 2) {
        printf("Radix Sort at bit position %d\n", pos);
        mark<T><<<numBlocks, threadsPerBlock>>>(a, flags, pos, size);

        printf("Printing flags after marking\n");
        //printFlags(flags, size);

        Scan::scan<uint4, uint4Add>(flags, size);

        printf("Printing flags after scanning\n");
        //printFlags(flags, size);

        uint4 totalOffset = {0, 0, 0, 0};
        uint4* lastFlag = flags + (size - 1);
        cudaMemcpy(&totalOffset, lastFlag, sizeof(uint4), cudaMemcpyDeviceToHost);
        totalOffset.w = totalOffset.z + totalOffset.y + totalOffset.x;
        totalOffset.z = totalOffset.y + totalOffset.x;
        totalOffset.y = totalOffset.x;
        totalOffset.x = 0;

        printf("Total Offset %d %d %d %d\n", totalOffset.x, totalOffset.y, totalOffset.z, totalOffset.w);

        scatter<T><<<numBlocks, threadsPerBlock>>>(a, b, flags, totalOffset, pos, size);
        //printAddress(b, size);

        T* temp = a;
        a = b;
        b = temp;
    }

    return a;
}

#endif
