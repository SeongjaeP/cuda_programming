# 🚀 CUDA Programming 30일 완성 플랜

CUDA 병렬 프로그래밍을 30일 동안 실습과 함께 완성하기 위한 학습 프로젝트입니다.  
GPU 아키텍처 이해부터 shared memory, 성능 최적화, 그리고 병렬 알고리즘까지 실무 수준의 역량을 갖추는 것을 목표로 합니다.

---

## 📌 목표

- CUDA 병렬 프로그래밍 모델 이해
- GPU 메모리 계층 및 실행 구조 체득
- 성능 최적화 실습 (memory, compute, occupancy 등)
- 병렬 알고리즘 구현 (reduction, scan, histogram 등)
- 실전 프로젝트로 포트폴리오 구성

---

## 📅 CUDA 30일 학습 로드맵

### ✅ Week 1: CUDA 기본기 익히기

| Day | Topic | Description |
|-----|-------|-------------|
| 1 | CUDA 환경 설정 | `nvcc`, GPU 확인, 첫 컴파일 |
| 2 | CUDA 아키텍처 개념 | Grid / Block / Thread 구조 |
| 3 | 커널 함수 이해 | 기본 벡터 덧셈 커널 작성 |
| 4 | 메모리 계층 개요 | global vs shared memory 실습 |
| 5 | Thread Indexing | 1D/2D thread ID 계산 |
| 6 | 시간 측정 | `cudaEvent_t` 활용 |
| 7 | 🧪 Mini Project ① | 반복 연산 벤치마크 구현 |

---

### ✅ Week 2: 메모리 최적화

| Day | Topic | Description |
|-----|-------|-------------|
| 8 | Coalesced Access | 메모리 접근 패턴 실험 |
| 9 | Shared Memory 활용 | 재사용 및 속도 비교 실습 |
| 10 | Bank Conflict 실험 | conflict 유도 및 해소 |
| 11 | Loop Unrolling | `#pragma unroll` 성능 비교 |
| 12 | Dynamic Shared Memory | `extern __shared__` 실습 |
| 13 | Constant & Texture Memory | CUDA memory 종류 비교 |
| 14 | 🧪 Mini Project ② | 다양한 memory 방식 성능 비교 |

---

### ✅ Week 3: 병렬 알고리즘 패턴

| Day | Topic | Description |
|-----|-------|-------------|
| 15 | Reduction | 공유 메모리로 합계 구현 |
| 16 | Scan (Prefix Sum) | Blelloch, Hillis-Steele |
| 17 | Histogram | Atomic 연산 실습 |
| 18 | Bitmask & Compaction | flag 기반 압축 |
| 19 | Warp-level Primitive | `__shfl_sync`, lane 통신 |
| 20 | Warp Divergence | 조건 분기 최적화 |
| 21 | 🧪 Mini Project ③ | 평균, 분산, max 병렬 계산기 |

---

### ✅ Week 4: 성능 분석 및 실전 최적화

| Day | Topic | Description |
|-----|-------|-------------|
| 22 | Occupancy & Register Pressure | `--ptxas-options=-v` 분석 |
| 23 | CUDA Profiler 사용법 | `nvprof`, `nsight`, `nsys` |
| 24 | 비동기 메모리 전송 | Pinned Memory & `cudaMemcpyAsync` |
| 25 | CUDA Stream 병렬화 | kernel overlap 실험 |
| 26 | Kernel Fusion | launch overhead 줄이기 |
| 27 | CUDA Graph (optional) | graph API 실습 |
| 28~30 | 🧪 Final Project | 실전 최적화 프로젝트 구현 |

---

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#what-is-the-cuda-c-programming-guide
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
