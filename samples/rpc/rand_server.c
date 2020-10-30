#include <stdlib.h>
#include "rand.h"

void* initialize_random_1_svc(long *argp, struct svc_req *rqstp) {
    static char *result;
    srand48(*argp);
    result = (void *) NULL;
    return (void *) &result;
}

double* get_next_random_1_svc(void *argp, struct svc_req *rqstp) {
    static double result;
    result = drand48();
    return (&result);
}

