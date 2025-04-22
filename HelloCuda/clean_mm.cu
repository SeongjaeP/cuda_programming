#include <iostream>
#include <cuda_runtime.h>

#define N 16
#define M 16
#define K 16
#define TILE_SIZE 8

__global__ void matmul_shared(const float* A, const float* B, float* C, int N, int K, int M) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < N && t * TILE_SIZE + threadIdx.x < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < M && t * TILE_SIZE + threadIdx.y < K)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * M + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            value += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < M) {
        C[row * M + col] = value;
    }
}

int main() {
    size_t size_A = N * K * sizeof(float);
    size_t size_B = K * M * sizeof(float);
    size_t size_C = N * M * sizeof(float);

    float* h_A = new float[N * K];
    float* h_B = new float[K * M];
    float* h_C = new float[N * M];

    for (int i = 0; i < N * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * M; i++) h_B[i] = 2.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_shared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N, K, M);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    std::cout << "result:\n";
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << h_C[i * M + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    return 0;
}
