#include <iostream>
#include <cuda_runtime.h>

#define N 16
#define M 16
#define TILE_SIZE 8

__global__ void matrix_add_shared(const float* A, const float* B, float* C, int width, int height){
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;

    // global → shared memory 로 복사
    if (row < height && col < width) {
        tile_A[ty][tx] = A[row * width + col];
        tile_B[ty][tx] = B[row * width + col];
    }

    // thread 
    __syncthreads();

    // shared memory에서 계산
    if (row < height && col < width){
        C[row * width + col] = tile_A[ty][tx] + tile_B[ty][tx];
    }
}

int main(){
    int size = N * M * sizeof(float);
    float* h_A = new float[N*M];
    float* h_B = new float[N*M];
    float* h_C = new float[N*M];

    for (int i = 0; i < N * M; i++){
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 threadPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matrix_add_shared<<<numBlocks, threadPerBlock>>>(d_A, d_B, d_C, M, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "shared memory result: " << std::endl;
    for (int i = 0; i < 5; i++){
        for (int j = 0; j < 5; j++){
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