#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "matmul.cuh"

float mat_a[MAT_SIZE][MAT_SIZE];
float mat_b[MAT_SIZE][MAT_SIZE];
float mat_c[MAT_SIZE][MAT_SIZE];

static void generate_matrices();
static void print_matrix(const char* path, const float mat[][MAT_SIZE]);

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s BLOCK_SIZE\n", argv[0]);
        return 1;
    }

    const int BLOCK_DIM = atoi(argv[1]);

    generate_matrices();
    print_matrix("mat_a.txt", mat_a);
    print_matrix("mat_b.txt", mat_b);

    matmul_global(BLOCK_DIM, mat_a, mat_b, mat_c);
    print_matrix("matmul_global.txt", mat_c);

    matmul_shared(BLOCK_DIM, mat_a, mat_b, mat_c);
    print_matrix("matmul_shared.txt", mat_c);

    matmul_optimized(BLOCK_DIM, mat_a, mat_b, mat_c);
    print_matrix("matmul_optimized.txt", mat_c);

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

static void print_matrix(const char* path, const float mat[][MAT_SIZE]) {
    FILE* fp;
    if ((fp = fopen(path, "w")) == NULL) {
        fprintf(stderr, "Error opening file %s\n", path);
        return;
    }

    for (int i = 0; i < MAT_SIZE; i++) {
        for (int j = 0; j < MAT_SIZE; j++) {
            fprintf(fp, "%f ", mat[i][j]);
        }
        fprintf(fp, "\n");
    }

    if (fclose(fp) != 0) {
        fprintf(stderr, "Error closing file %s\n", path);
        return;
    }
}

