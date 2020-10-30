# Sample RPC program

## Steps
1. Create the `*.x` file
2. Generate files using `rpcgen`
  ```bash
  rpcgen -C -a rand.x
  ```
3. Fill in the `*_client.c` and `*_server.c` files
4. Compile using the generated `Makefile.*`
  ```bash
  makefile -f Makefile.rand
  ```
5. Run server and client programs
  ```bash
  # in cspro2
  ./rand_server

  # in cspro1
  ./rand_client cspro2.sogang.ac.kr 20 10
  ```

### Sample `rand_client.c`
```c
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
```

### Sample `rand_server.c`
```c
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
```

