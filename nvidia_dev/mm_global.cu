#include <stdio.h>

typedef struct {
    int width;
    int height;
    float *element;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matmul - Host code
// Matrix dimension are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C) {

    // Load A and B device memory
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.element, size);
    cudaMemcpy(d_A.element, A.element, size, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size_t size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.element, size);
    cudaMemcpy(d_B.element, B.element, size, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.element, size);

    //Invoke Kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.element, d_C.element, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.element);
    cudaFree(d_B.element);
    cudaFree(d_C.element);
}


// Matmul kernel called by Matmul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e; e < A.width; e++) {
        Cvalue += A.element[row * A.width + e] * B.element[e * B.width + col];
        C.element[row*C.width + col];
    }
}