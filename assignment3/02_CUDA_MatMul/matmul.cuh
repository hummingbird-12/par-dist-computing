#define MAT_SIZE 4

__host__ void matmul_global(const int BLOCK_DIM, const float mat_a[][MAT_SIZE], const float mat_b[][MAT_SIZE], float mat_c[][MAT_SIZE]);

