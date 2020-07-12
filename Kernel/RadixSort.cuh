#ifndef RADIX_SORT_CUH
#define RADIX_SORT_CUH

#include "Scan.cuh"

namespace RadixSort {
    template<typename T>
    T* sort(T* a, T* b, uint4* flags_a, uint4* flags_b, int size);
}

namespace {

__device__
uint4 uint4Add(uint4 a, uint4 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

__device__
int uint4Dot(uint4 a, uint4 b) {
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z) + (a.w * b.w);
}

template<typename T>
__global__
void mark(const T* elements, uint4* flags, int pos, int size) {

    int tid = threadIdx.x;
    int globalOffset = blockDim.x * blockIdx.x;
    int gid = globalOffset + tid;

    if (gid >= size) { return; }

    T mask = 3 << pos;
    T index = (elements[gid] & mask) >> pos;
    
    uint4 flag = {0, 0, 0, 0};
    //flag[index] = 1;
    
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
    
    uint4 address;
    if (tid == 0) {
        address = {0, 0, 0, 0};
    } else {
        address = addresses[gid - 1];
    }

    address = uint4Add(address, totalOffset);

    T mask = 3 << pos;
    T index = (value & mask) >> pos;
    uint4 flag = {0, 0, 0, 0};
    //flag[index] = 1;

    out[uint4Dot(flag, address)] = value;
}

} // namespace anonymous

template<typename T>
T* RadixSort::sort(T* a, T* b, uint4* flags_a, uint4* flags_b, int size) {
    constexpr int numBits = sizeof(T) * 8;
    constexpr int threadsPerBlock = 1024;
    const int numBlocks = ceil(size / (float) threadsPerBlock);

    for (int pos = 0; pos < numBits; pos += 2) {

        mark<T><<<numBlocks, threadsPerBlock>>>(a, flags_a, pos, size);
        
        //Scan::scan<uint4, uint4Add>(flags_a, flags_a, flags_b, size);
        uint4 totalOffset = {0, 0, 0, 0};
        
        scatter<T><<<numBlocks, threadsPerBlock>>>(a, b, flags_a, totalOffset, pos, size);

        T* temp = a;
        a = b;
        b = temp;
    }

    return a;
}

#endif
