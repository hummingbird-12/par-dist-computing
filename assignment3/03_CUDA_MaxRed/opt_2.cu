#include <math.h>
#include <stdio.h>

#include "max_reduction.cuh"

__global__ static void kernel(int* arr, int* n, int* stride);

__host__ void reduction_opt_2(const int* arr, const int n, const int BLOCK_SIZE) {
    const int SIZE = sizeof(int) * n;
    int mx;
    int *dev_arr, *dev_n, *dev_stride;
    double start, end;

    const int GRID_DIM = ceil((float) n / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    dim3 grid(GRID_DIM);

    GET_TIME(start);

    cudaMalloc((void**) &dev_arr, SIZE);
    cudaMalloc((void**) &dev_n, sizeof(int));
    cudaMalloc((void**) &dev_stride, sizeof(int));

    cudaMemcpy(dev_arr, arr, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_n, &n, sizeof(int), cudaMemcpyHostToDevice);

    for (int i = n, stride = (n + 1) / 2;
         i >= 1;
         i /= 2, stride = (stride + 1) / 2) {
        cudaMemcpy(dev_stride, &stride, sizeof(int), cudaMemcpyHostToDevice);
        kernel<<<grid, block, SIZE>>>(dev_arr, dev_n, dev_stride);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(&mx, dev_arr, sizeof(int), cudaMemcpyDeviceToHost);

    GET_TIME(end);

    printf("[reduction_opt_2]\tMaximum: %d\tTime: %fs\n", mx, end - start);

    cudaFree(dev_arr);
    cudaFree(dev_n);
    cudaFree(dev_stride);
}

__global__ static void kernel(int* arr, int* n, int* stride) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int shared[];

    if (tid < *stride && tid + *stride < *n) {
        shared[tid] = arr[tid];
        shared[tid + *stride] = arr[tid + *stride];
        arr[tid] = max(shared[tid], shared[tid + *stride]);
    }
}
