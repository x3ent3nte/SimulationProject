#ifndef REDUCE_CUH
#define REDUCE_CUH

namespace Reduce {
    
    template<typename T, T (*FN)(T, T)>
    int reduce(T* d_a, T* d_b, int size);
}

namespace {

template<typename T, T (*FN)(T, T), int blockSize>
__device__
void lastWarpReduce(volatile T* sharedInts, int tid, int offset) {
    if (blockSize >= 64) {
        if (tid < offset) { sharedInts[tid] = FN(sharedInts[tid], sharedInts[tid + offset]); }
        offset >>= 1;
    }
    
    if (blockSize >= 32) {
        if (tid < offset) { sharedInts[tid] = FN(sharedInts[tid], sharedInts[tid + offset]); }
        offset >>= 1;
    }
   
    if (blockSize >= 16) {
        if (tid < offset) { sharedInts[tid] = FN(sharedInts[tid], sharedInts[tid + offset]); }
        offset >>= 1;
    }
    
    if (blockSize >= 8) {
        if (tid < offset) { sharedInts[tid] = FN(sharedInts[tid], sharedInts[tid + offset]); }
        offset >>= 1;
    }
    
    if (blockSize >= 4) {
        if (tid < offset) { sharedInts[tid] = FN(sharedInts[tid], sharedInts[tid + offset]); }
        offset >>= 1;
    }
    
    if (blockSize >= 2) {
        if (tid < offset) { sharedInts[tid] = FN(sharedInts[tid], sharedInts[tid + offset]); }
    }
}

template<typename T, T (*FN)(T, T), int blockSize>
__global__
void reduceKernel(T* in, T* out, int size) {
    __shared__ T sharedInts[blockSize * sizeof(T)];

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

    int offset = min(blockDim.x, size - blockStart) >> 1;
    /*for (; offset > 32; offset >>= 1) {
        if (tid < offset) {
            sharedInts[tid] = FN(sharedInts[tid], sharedInts[tid + offset]);
        }
        __syncthreads();
    }*/
    
    if (blockSize >= 1024) {
        if (tid < offset) { sharedInts[tid] = FN(sharedInts[tid], sharedInts[tid + offset]); }
        __syncthreads();
        offset >>= 1;
    }

    if (blockSize >= 512) {
        if (tid < offset) { sharedInts[tid] = FN(sharedInts[tid], sharedInts[tid + offset]); }
        __syncthreads();
        offset >>= 1;
    }

    if (blockSize >= 256) {
        if (tid < offset) { sharedInts[tid] = FN(sharedInts[tid], sharedInts[tid + offset]); }
        __syncthreads();
        offset >>= 1;
    }

    if (blockSize >= 128) {
        if (tid < offset) { sharedInts[tid] = FN(sharedInts[tid], sharedInts[tid + offset]); }
        __syncthreads();
        offset >>= 1;
    }

    if (tid < 32) { lastWarpReduce<T, FN, blockSize>(sharedInts, tid, offset); }

    if (tid == 0) {
        out[blockIdx.x] = sharedInts[0];
    }
}

} // namespace anonymous

template<typename T, T (*FN)(T, T)>
int Reduce::reduce(T* d_a, T* d_b, int size) {
    
    constexpr int threadsPerBlock = 128;
    
    while (size > 1) {
        int numBlocks = ceil(size / ((float) threadsPerBlock * 2));
        reduceKernel<T, FN, threadsPerBlock><<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(d_a, d_b, size);
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

#endif
