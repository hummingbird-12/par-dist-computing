#ifndef _MAX_REDUCTION_CUH_INCLUDED_
#define _MAX_REDUCTION_CUH_INCLUDED_

#include <sys/time.h>

#define GET_TIME(now)                           \
    {                                           \
        struct timeval t;                       \
        gettimeofday(&t, NULL);                 \
        now = t.tv_sec + t.tv_usec / 1000000.0; \
    }

#define DEF_ARR_SIZE 10000
#define DEF_BLOCK_SIZE 512

__host__ void reduction_divergent(const int* arr, const int n);
__host__ void reduction_opt_1(const int* arr, const int n);
__host__ void reduction_opt_2(const int* arr, const int n,
                              const int BLOCK_SIZE);

#endif /* _MAX_REDUCTION_CUH_INCLUDED_ */
