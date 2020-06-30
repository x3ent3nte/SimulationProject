#ifndef SCAN_CUH
#define SCAN_CUH

namespace Scan {
    template<typename T, T (*FN)(T, T)>
    void scan(T* in, T* out, T* offsets, int size);
}

namespace {

template<typename T, T (*FN)(T, T)>
__global__
void addBlockOffsets(T* ints, T* offsets, int size) {
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

    extern __shared__ int sharedInts[];

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

} // namespace anonymous

template<typename T, T (*FN)(T, T)>
void Scan::scan(T* in, T* out, T* offsets, int size) {
    constexpr int threadsPerBlock = 512;
    int numBlocks = ceil(size / float(threadsPerBlock));
    scanKernel<T, FN><<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(in, out, offsets, size);

    if (numBlocks > 1) { 
        int sizeOfOffsetAdd = size - threadsPerBlock;
        int numBlocksToAddOffsets = ceil(sizeOfOffsetAdd / (float) threadsPerBlock);
         
        // *Scan sum the offsets placeholder*
        
        addBlockOffsets<T, FN><<<numBlocksToAddOffsets, threadsPerBlock>>>(out + threadsPerBlock, offsets, sizeOfOffsetAdd);
    }
}

#endif
