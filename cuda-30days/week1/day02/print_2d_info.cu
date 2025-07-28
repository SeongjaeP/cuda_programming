#include <stdio.h>

__global__ void print2DInfo() {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    printf("[Block (%d,%d), Thread (%d,%d)]\n", bx, by, tx, ty);
}

int main() {
    dim3 blockDim(2, 2);
    dim3 gridDim(2, 2);

    print2DInfo<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
    return 0;
}