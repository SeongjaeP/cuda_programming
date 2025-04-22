#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 128
#define M 128
#define K 128

int main() {
    size_t size_A = N * K * sizeof(float);
    size_t size_B = K * M * sizeof(float);
    size_t size_C = N * M * sizeof(float);

    float *h_A = new float[N * K];
    float *h_B = new float[K * M];
    float *h_C = new float[N * M];

    for (int i = 0; i < N * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * M; i++) h_B[i] = 2.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // cuBLAS 초기화
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // 실행 시간 측정
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // cuBLAS 행렬 곱 (C = alpha * A * B + beta * C)
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha,
                d_B, M,
                d_A, K,
                &beta,
                d_C, M);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    std::cout << "cuBLAS execution time: " << milliseconds << " ms\n";

    float gflops = (2.0f * N * K * M) / (milliseconds / 1000.0f) / 1e9f;
    std::cout << "Estimated cuBLAS GFLOPS: " << gflops << std::endl;

    std::cout << "Sample result (5x5):" << std::endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            std::cout << h_C[i * M + j] << " ";
        }
        std::cout << std::endl;
    }

    // 정리
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    cublasDestroy(handle);

    return 0;
}
