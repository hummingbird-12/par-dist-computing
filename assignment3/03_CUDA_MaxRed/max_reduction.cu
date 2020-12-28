#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "max_reduction.cuh"

int* arr;

static void generate_integers(const int ARR_SIZE);
static int sequential(const int ARR_SIZE);

int main(int argc, char* argv[]) {
    if (argc != 2 && argc != 3) {
        fprintf(stderr, "Usage: %s BLOCK_SIZE [ARRAY_SIZE]\n", argv[0]);
        return 1;
    }

    const int BLOCK_SIZE = atoi(argv[1]);
    const int ARR_SIZE = (argc == 3 ? atoi(argv[2]) : DEF_ARR_SIZE);

    generate_integers(ARR_SIZE);
    sequential(ARR_SIZE);
    reduction_divergent(arr, ARR_SIZE);
    reduction_opt_1(arr, ARR_SIZE);
    reduction_opt_2(arr, ARR_SIZE, BLOCK_SIZE);

    free(arr);

    return 0;
}

static void generate_integers(const int ARR_SIZE) {
    if ((arr = (int*) malloc(sizeof(int) * ARR_SIZE)) == NULL) {
        fprintf(stderr, "Error while allocating memory\n");
        exit(1);
    }
    srand(time(NULL));
    for (int i = 0; i < ARR_SIZE; i++) {
        arr[i] = rand();
    }
}

static int sequential(const int ARR_SIZE) {
    double start, end;
    GET_TIME(start);
    int mx = arr[0];
    for (int i = 1; i < ARR_SIZE; i++) {
        mx = max(mx, arr[i]);
    }
    GET_TIME(end);
    printf("[reduction_sequential]\tMaximum: %d\tTime: %fs\n", mx, end - start);
    return mx;
}
