#include <stdio.h>
int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("CUDA-enabled GPU found: %d device(s)\n", deviceCount);
    return 0;
}
