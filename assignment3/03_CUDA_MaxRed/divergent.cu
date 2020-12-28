#include <math.h>
#include <stdio.h>

#include "max_reduction.cuh"

__global__ static void kernel(int arr[ARR_SIZE], int* stride);

__host__ void reduction_divergent(const int arr[ARR_SIZE]) {
    const int SIZE = sizeof(int) * ARR_SIZE;
    int mx, n, stride;
    int *dev_arr, *dev_stride;
    double start, end;

    const int GRID_DIM = ceil((float) ARR_SIZE / DEF_BLOCK_SIZE);
    dim3 block(DEF_BLOCK_SIZE);
    dim3 grid(GRID_DIM);

    GET_TIME(start);

    cudaMalloc((void**) &dev_arr, SIZE);
    cudaMalloc((void**) &dev_stride, sizeof(int));

    cudaMemcpy(dev_arr, arr, SIZE, cudaMemcpyHostToDevice);

    n = ARR_SIZE;
    stride = 1;
    while (n >= 1) {
        cudaMemcpy(dev_stride, &stride, sizeof(int), cudaMemcpyHostToDevice);
        kernel<<<grid, block>>>(dev_arr, dev_stride);
        cudaDeviceSynchronize();

        stride *= 2;
        n /= 2;
    }

    cudaMemcpy(&mx, dev_arr, sizeof(int), cudaMemcpyDeviceToHost);

    GET_TIME(end);

    printf("[reduction_divergent]\tMaximum: %d\tTime: %fs\n", mx, end - start);

    cudaFree(dev_arr);
    cudaFree(dev_stride);
}

__global__ static void kernel(int arr[ARR_SIZE], int* stride) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid % (2 * (*stride)) == 0 && tid + (*stride) < ARR_SIZE) {
        arr[tid] = max(arr[tid], arr[tid + (*stride)]);
    }
}
