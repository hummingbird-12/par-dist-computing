#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "matmul.cuh"

float mat_a[MAT_SIZE][MAT_SIZE];
float mat_b[MAT_SIZE][MAT_SIZE];
float mat_c[MAT_SIZE][MAT_SIZE];

static void generate_matrices();

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s BLOCK_SIZE\n", argv[0]);
        return 1;
    }

    const int BLOCK_SIZE = atoi(argv[1]);

    generate_matrices();

    matmul_global(BLOCK_SIZE, mat_a, mat_b, mat_c);
    matmul_shared(BLOCK_SIZE, mat_a, mat_b, mat_c);
    matmul_optimized(BLOCK_SIZE, mat_a, mat_b, mat_c);

    return 0;
}

static void generate_matrices() {
    srand(time(NULL));
    for (int i = 0; i < MAT_SIZE; i++) {
        for (int j = 0; j < MAT_SIZE; j++) {
            mat_a[i][j] = (float) (rand() - rand()) / RAND_MAX;
            mat_b[i][j] = (float) (rand() - rand()) / RAND_MAX;
        }
    }
}
