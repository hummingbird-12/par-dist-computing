#ifndef _MATMUL_CUH_INCLUDED_
#define _MATMUL_CUH_INCLUDED_

#include <sys/time.h>

#define GET_TIME(now)                           \
    {                                           \
        struct timeval t;                       \
        gettimeofday(&t, NULL);                 \
        now = t.tv_sec + t.tv_usec / 1000000.0; \
    }

#define MAT_SIZE 4096

__host__ void matmul_global(const int BLOCK_SIZE, const float mat_a[][MAT_SIZE],
                            const float mat_b[][MAT_SIZE],
                            float mat_c[][MAT_SIZE]);
__host__ void matmul_shared(const int BLOCK_SIZE, const float mat_a[][MAT_SIZE],
                            const float mat_b[][MAT_SIZE],
                            float mat_c[][MAT_SIZE]);
__host__ void matmul_optimized(const int BLOCK_SIZE,
                               const float mat_a[][MAT_SIZE],
                               const float mat_b[][MAT_SIZE],
                               float mat_c[][MAT_SIZE]);

#endif /* _MATMUL_CUH_INCLUDED_ */
