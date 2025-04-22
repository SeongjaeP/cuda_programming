// ctrl + i -> 코파일럿

#include <iostream>
#include <cuda_runtime.h>


__global__ void vector_add(const float* A, const float* B, float* C, int B) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 256;
    size_t size = N * sizeof(float);

    float *h_A =  new float[N];
    float *h_B =  new float[N];
    float *h_C =  new float[N];

    for (int i = 0; i < N; i++){
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size ,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size ,cudaMemcpyHostToDevice);

    vector_add<<<1, N>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++){
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}