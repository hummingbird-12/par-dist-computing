#include <stdio.h>

#include "matmul.cuh"

__global__ static void kernel(float* mat_a, float* mat_b, float* mat_c);

__host__ void matmul_global(const int BLOCK_SIZE, const float mat_a[][MAT_SIZE],
                            const float mat_b[][MAT_SIZE],
                            float mat_c[][MAT_SIZE]) {
    const int SIZE = sizeof(float) * MAT_SIZE * MAT_SIZE;
    float *dev_a, *dev_b, *dev_c;
    double start, end;

    const int GRID_DIM = MAT_SIZE / BLOCK_SIZE;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(GRID_DIM, GRID_DIM);

    GET_TIME(start);

    cudaMalloc((void**) &dev_a, SIZE);
    cudaMalloc((void**) &dev_b, SIZE);
    cudaMalloc((void**) &dev_c, SIZE);

    cudaMemcpy(dev_a, mat_a, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, mat_b, SIZE, cudaMemcpyHostToDevice);
    cudaMemset(dev_c, 0, SIZE);

    kernel<<<grid, block>>>(dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();

    cudaMemcpy(mat_c, dev_c, SIZE, cudaMemcpyDeviceToHost);

    GET_TIME(end);

    printf("[matmul_global]\tTime: %fs\n", end - start);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

__global__ static void kernel(float* mat_a, float* mat_b, float* mat_c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < MAT_SIZE; i++) {
        mat_c[row * MAT_SIZE + col] +=
            mat_a[row * MAT_SIZE + i] * mat_b[i * MAT_SIZE + col];
    }
}
