#include <stdlib.h>
#include <stdio.h>
#include "rand.h"

void main(int argc, char *argv[ ]) {
    int iters,i;
    long myseed;
    CLIENT *clnt;
    void *result_1;
    double *result_2;
    char *arg;

    if ( argc != 4 ) {
        fprintf(stderr, "Usage: %s host seed iterations\n", argv[0]);
        exit(1);
    }
    clnt = clnt_create(argv[1], RAND_PROG, RAND_VERS, "udp"); 

    if ( clnt == (CLIENT *) NULL ) {
        clnt_pcreateerror(argv[1]);
        exit(1);
    }

    myseed = (long) atoi (argv[2]);
    iters = atoi (argv[3]);

    result_1 = initialize_random_1(&myseed, clnt);

    if (result_1 == (void *) NULL) {
        clnt_perror(clnt, "call failed");
    }

    for( i=0; i<iters; i++) {
        result_2 = get_next_random_1((void *)&arg, clnt);
        if (result_2 == (double *) NULL) {
            clnt_perror(clnt, "call failed");
        }
        else {
            printf("%d : %f\n", i, *result_2);
        }
    }

    clnt_destroy(clnt);
    exit(0);
}

