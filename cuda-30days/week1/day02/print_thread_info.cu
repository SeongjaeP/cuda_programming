#include <stdio.h>

__global__ void printThreadInfo() {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim = blockDim.x;
    int gid = bid * bdim + tid;

    printf("[Block %d, Thread %d] → Global ID: %d\n", bid, tid, gid);
}

int main() {
    int threadsPerBlock = 4;
    int numBlocks = 3;

    printThreadInfo<<<numBlocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();

    return 0;
}