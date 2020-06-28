#include <iostream>

#define SIZE 1024

__global__
void vectorAdd(int* a, int* b, int* c, int size) {
    int i = (blockIdx.x * SIZE) + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    
    int* a = (int*) malloc(SIZE * SIZE * sizeof(int));
    int* b = (int*) malloc(SIZE * SIZE * sizeof(int));
    int* c = (int*) malloc(SIZE * SIZE * sizeof(int));

    int* d_a;
    int* d_b;
    int* d_c;

    cudaMalloc(&d_a, SIZE * SIZE * sizeof(int));
    cudaMalloc(&d_b, SIZE * SIZE * sizeof(int));
    cudaMalloc(&d_c, SIZE * SIZE * sizeof(int));

    for (int i = 0; i < SIZE * SIZE; ++i) {
        a[i] = i;
        b[i] = i;
        c[i] = 0;
    }

    cudaMemcpy(d_a, a, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);

    vectorAdd<<<SIZE, SIZE>>>(d_a, d_b, d_c, SIZE * SIZE);

    cudaMemcpy(c, d_c, SIZE * SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE * SIZE; ++i) {
        std::cout << c[i] << "\n";
    }

    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
