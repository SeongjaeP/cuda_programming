#include <iostream>
#include <cuda_runtime.h>

#define N 16 // row size
#define M 16 // column size

__global__ void matrix_add(const float* A, const float* B, float* C, int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}


int main() {
    int size = N * M * sizeof(float);

    float* h_A = new float[N * M];
    float* h_B = new float[N * M];
    float* h_C = new float[N * M];

    for (int i = 0; i < N * M; i++){
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadPerBlock(8, 8);
    dim3 numBlock((M + 7) / 8, (N + 7) / 8);

    matrix_add<<<numBlock, threadPerBlock>>>(d_A, d_B, d_C, M, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Result: " << std::endl;
    for (int i = 0; i< 5; ++i){
        for (int j = 0; j < 5; ++j){
            std::cout << h_C[i * M + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}