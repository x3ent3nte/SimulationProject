#ifndef SCAN_CUH
#define SCAN_CUH

namespace Scan {
    template<typename T, T (*FN)(T, T)>
    void scan(T* in, T* out, int size);
}

namespace {

template<typename T, T (*FN)(T, T)>
__global__
void scanKernel(T* in,  T* out, int size) {
    int tid = threadIdx.x;
    int globalOffset = blockDim.x * blockIdx.x;
    int gid = globalOffset + tid;

    if (gid >= size) { return; }

    extern __shared__ int sharedInts[];

    int localSize = size - globalOffset;

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
}

} // namespace anonymous

template<typename T, T (*FN)(T, T)>
void Scan::scan(T* in, T* out, int size) {
    int threadsPerBlock = 1024;
    scanKernel<T, FN><<<1, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(in, out, threadsPerBlock);
}

#endif
