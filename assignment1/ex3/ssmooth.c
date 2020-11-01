#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ppm.h"
#include "utils.h"

#define FNAME_MOD "_ssmooth"

int errno;

int main(int argc, char** argv) {
    if (argc != 2) {
        errno = EINVAL;
        perror("Error while parsing arguments");
        exit(1);
    }

    clock_t start, end;

    printf("===== Image smoothening using sequential processing =====\n");
    start = clock();

    const char* fname = argv[1];
    PPM* image = read_ppm(fname);
    PPM* trans = create_mod_ppm(image, FNAME_MOD);
    const int rows = image->rows;
    const int cols = image->cols;

    for (int i = 0; i < rows * cols; i++) {
        smooth(image->data, trans->data, rows, cols, i);
    }

    write_ppm(trans);

    end = clock();
    printf("Elapsed time: %.3lfms\n", (end - start) * 1000.0 / CLOCKS_PER_SEC);

    free_ppm(image);
    free_ppm(trans);

    return 0;
}
