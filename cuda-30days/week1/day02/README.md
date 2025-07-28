# Day 2 – CUDA 아키텍처 개념 (Grid / Block / Thread)

## ✅ 학습 내용 요약

- CUDA의 실행 단위: Thread, Block, Grid 구조 이해
- 각 Thread의 고유 ID 계산 방식 (`global_id = blockIdx.x * blockDim.x + threadIdx.x`)
- `dim3`를 활용한 2D Grid/Block 구성법 학습

## 📂 포함된 파일

| 파일명 | 설명 |
|--------|------|
| `print_thread_info.cu` | 1D Grid/Block 구조에서 thread index 출력 |
| `print_2d_info.cu`     | 2D Grid/Block 구조 출력 |

## 🧱 CUDA Thread-Hierarchy 구조

![CUDA Grid-Block-Thread 구조](./images/cuda_grid_block_thread.png)

- **Grid**는 여러 개의 Block으로 구성됨
- **Block**은 여러 Thread를 포함하고 있으며 **Block 내부의 Thread끼리 공유 메모리 공유**
- **Thread**는 `threadIdx` 기반으로 개별 실행 단위
