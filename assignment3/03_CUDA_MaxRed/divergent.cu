#include <math.h>
#include <stdio.h>

#include "max_reduction.cuh"

__global__ static void kernel(int arr[ARR_SIZE], int mx);

__host__ void reduction_divergent(const int arr[ARR_SIZE]) {
    const int SIZE = sizeof(int) * ARR_SIZE;
    int *dev_arr, *dev_mx, mx;
    double start, end;

    const int GRID_DIM = ceil((float) ARR_SIZE / DEF_BLOCK_SIZE);
    dim3 block(DEF_BLOCK_SIZE);
    dim3 grid(GRID_DIM);

    GET_TIME(start);

    cudaMalloc((void**) &dev_arr, SIZE);
    cudaMalloc((void**) &dev_mx, sizeof(int));

    cudaMemcpy(dev_arr, arr, SIZE, cudaMemcpyHostToDevice);
    cudaMemset(dev_mx, 0, sizeof(int));

    kernel<<<grid, block>>>(dev_arr, dev_mx);
    cudaDeviceSynchronize();

    cudaMemcpy(&mx, dev_mx, sizeof(int), cudaMemcpyDeviceToHost);

    GET_TIME(end);

    printf("[reduction_divergent] Maximum: %d\tTime: %fs\n", mx, end - start);

    cudaFree(dev_arr);
    cudaFree(dev_mx);
}

__global__ static void kernel(int arr[ARR_SIZE], int mx) {
    while (n > 1) {
	    int stride = (n + 1) / 2; // round up to find memory offset
	    int srcIdx = i + stride;
	    if (srcIdx < n) {
		    arr[i] += arr[srcIdx];
        }
	    barrier(CLK_GLOBAL_MEM_FENCE); /* subtle: needed so we can read newly added values */
	    n = stride; /* new size is everything except what we've already read */
   }
}
