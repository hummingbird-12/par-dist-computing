#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "max_reduction.cuh"

int arr[ARR_SIZE];

static void generate_integers();
static int sequential();

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s BLOCK_SIZE\n", argv[0]);
        return 1;
    }

    const int BLOCK_DIM = atoi(argv[1]);

    generate_integers();
    sequential();

    return 0;
}

static void generate_integers() {
    srand(time(NULL));
    for (int i = 0; i < ARR_SIZE; i++) {
        arr[i] = rand();
    }
}

static int sequential() {
    double start, end;
    GET_TIME(start);
    int mx = arr[0];
    for (int i = 1; i < ARR_SIZE; i++) {
        mx = max(mx, arr[i]);
    }
    GET_TIME(end);
    printf("[sequential] Maximum: %d\tTime: %fs\n", mx, end - start);
    return mx;
}
