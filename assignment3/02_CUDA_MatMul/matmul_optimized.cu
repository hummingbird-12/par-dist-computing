#include <stdio.h>

#include "matmul.cuh"

__global__ static void kernel(float* mat_a, float* mat_b, float* mat_c);

__host__ void matmul_optimized(const int BLOCK_DIM,
                               const float mat_a[][MAT_SIZE],
                               const float mat_b[][MAT_SIZE],
                               float mat_c[][MAT_SIZE]) {
    const int SIZE = sizeof(float) * MAT_SIZE * MAT_SIZE;
    float *dev_a, *dev_b, *dev_c;
    double start, end;

    const int GRID_DIM = MAT_SIZE / BLOCK_DIM;
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(GRID_DIM, GRID_DIM);

    GET_TIME(start);

    cudaMalloc((void**) &dev_a, SIZE);
    cudaMalloc((void**) &dev_b, SIZE);
    cudaMalloc((void**) &dev_c, SIZE);

    cudaMemcpy(dev_a, mat_a, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, mat_b, SIZE, cudaMemcpyHostToDevice);
    cudaMemset(dev_c, 0, SIZE);

    kernel<<<grid, block,
             sizeof(float) * BLOCK_DIM * BLOCK_DIM*(2 * GRID_DIM + 1)>>>(
        dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();

    cudaMemcpy(mat_c, dev_c, SIZE, cudaMemcpyDeviceToHost);

    GET_TIME(end);

    printf("[matmul_optimized] Time: %fs\n", end - start);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

__global__ static void kernel(float* mat_a, float* mat_b, float* mat_c) {
    extern __shared__ float shared[];
    float* shared_a = shared;  // [MAT_SIZE][blockDim.y]
    float* shared_b =
        shared_a + MAT_SIZE * blockDim.y;  // [MAT_SIZE][blockDim.x]
    float* shared_c =
        shared_b + MAT_SIZE * blockDim.x;  // [blockDim.y][blockDim.x]

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x * blockDim.x + threadIdx.y;

    shared_c[tid] = 0.0f;
    for (int i = 0; i < MAT_SIZE; i++) {
        shared_a[i * blockDim.y + threadIdx.y] = mat_a[row * MAT_SIZE + i];
        shared_b[i * blockDim.x + threadIdx.x] = mat_b[i * MAT_SIZE + col];
        shared_c[tid] += shared_a[i * blockDim.y + threadIdx.y] *
                         shared_b[i * blockDim.x + threadIdx.x];
    }
    mat_c[row * MAT_SIZE + col] = shared_c[tid];
}
