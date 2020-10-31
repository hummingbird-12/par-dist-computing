#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include "ppm.h"
#include "utils.h"

#define FNAME_MOD "_flip"

int errno;

int main(int argc, char** argv) {
    if (argc != 2) {
        errno = EINVAL;
        perror("Error while parsing arguments");
        exit(1);
    }

    char* fname = argv[1];
    PPM* image = read_ppm(fname);
    PPM* trans = create_mod_ppm(image, FNAME_MOD);
    const int rows = image->rows;
    const int cols = image->cols;

    for (int i = 0; i < rows; i++) {
        flip(image->data + i * cols, trans->data + i * cols, cols);
    }

    write_ppm(trans);

    free_ppm(image);
    free_ppm(trans);

    return 0;
}
