#include <stdio.h>

__global__ void add(int *a, int *b, int *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    const int N = 10;
    int h_a[N], h_b[N], h_c[N];
    int *d_a, *d_b, *d_c;

    for (int i = 0; i < N; i++){
        h_a[i] = i;
        h_b[i] = i * i;
    }

    size_t size = N * sizeof(int);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    add<<<1, N>>>(d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++){
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}