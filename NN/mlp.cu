#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define N 4   // Batch size
#define K 3   // Input dim
#define H 5   // Hidden dim
#define M 2   // Output dim

// Linear forward: Y = XW + B
__global__ void linear_forward(const float* X, const float* W, const float* B, float* Y, int in_dim, int out_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < out_dim) {
        float sum = 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            sum += X[row * in_dim + i] * W[i * out_dim + col];
        }
        Y[row * out_dim + col] = sum + B[col];
    }
}

// ReLU
__global__ void relu_forward(const float* X, float* Y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        Y[idx] = X[idx] > 0 ? X[idx] : 0.0f;
    }
}

// Softmax (row-wise)
__global__ void softmax_forward(float* logits, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float max_val = logits[row * cols];
    for (int i = 1; i < cols; ++i)
        if (logits[row * cols + i] > max_val)
            max_val = logits[row * cols + i];

    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
        output[row * cols + i] = expf(logits[row * cols + i] - max_val);
        sum += output[row * cols + i];
    }

    for (int i = 0; i < cols; ++i) {
        output[row * cols + i] /= sum;
    }
}

int main() {
    float h_X[N * K] = {1,2,3,4,5,6,7,8,9,10,11,12};
    float h_W1[K * H], h_B1[H], h_W2[H * M], h_B2[M];

    for (int i = 0; i < K * H; i++) h_W1[i] = 0.1f;
    for (int i = 0; i < H; i++) h_B1[i] = 0.1f;
    for (int i = 0; i < H * M; i++) h_W2[i] = 0.2f;
    for (int i = 0; i < M; i++) h_B2[i] = 0.2f;

    float *d_X, *d_W1, *d_B1, *d_Z1, *d_A1;
    float *d_W2, *d_B2, *d_Z2, *d_A2;
    cudaMalloc(&d_X, sizeof(float) * N * K);
    cudaMalloc(&d_W1, sizeof(float) * K * H);
    cudaMalloc(&d_B1, sizeof(float) * H);
    cudaMalloc(&d_Z1, sizeof(float) * N * H);
    cudaMalloc(&d_A1, sizeof(float) * N * H);
    cudaMalloc(&d_W2, sizeof(float) * H * M);
    cudaMalloc(&d_B2, sizeof(float) * M);
    cudaMalloc(&d_Z2, sizeof(float) * N * M);
    cudaMalloc(&d_A2, sizeof(float) * N * M);

    cudaMemcpy(d_X, h_X, sizeof(float) * N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, h_W1, sizeof(float) * K * H, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B1, h_B1, sizeof(float) * H, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, sizeof(float) * H * M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2, h_B2, sizeof(float) * M, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks1((H + 15) / 16, (N + 15) / 16);
    dim3 blocks2((M + 15) / 16, (N + 15) / 16);

    linear_forward<<<blocks1, threads>>>(d_X, d_W1, d_B1, d_Z1, K, H);
    relu_forward<<<(N*H+255)/256, 256>>>(d_Z1, d_A1, N*H);
    linear_forward<<<blocks2, threads>>>(d_A1, d_W2, d_B2, d_Z2, H, M);
    softmax_forward<<<(N+127)/128, 128>>>(d_Z2, d_A2, N, M);

    float h_A2[N * M];
    cudaMemcpy(h_A2, d_A2, sizeof(h_A2), cudaMemcpyDeviceToHost);

    std::cout << "\nSoftmax output:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            std::cout << h_A2[i * M + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_X); cudaFree(d_W1); cudaFree(d_B1); cudaFree(d_Z1); cudaFree(d_A1);
    cudaFree(d_W2); cudaFree(d_B2); cudaFree(d_Z2); cudaFree(d_A2);

    return 0;
}
