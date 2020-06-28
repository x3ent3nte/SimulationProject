#ifndef REDUCE_CUH
#define REDUCE_CUH

namespace Reduce {
    
    template<typename T, T (*FN)(T, T)>
    int reduce(T* d_a, T* d_b, int size);
}

namespace {

template<typename T, T (*FN)(T, T)>
__global__
void reduceKernel(T* in, T* out, int size) {
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

} // namespace anonymous

template<typename T, T (*FN)(T, T)>
int Reduce::reduce(T* d_a, T* d_b, int size) {
    int threadsPerBlock = 256;
    while (size > 1) {
        int numBlocks = ceil(size / ((float) threadsPerBlock * 2));
        reduceKernel<T, FN><<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(d_a, d_b, size);
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
