#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi.h"
#include "ppm.h"
#include "utils.h"

#define FNAME_MOD "_psmooth"

int errno;

int main(int argc, char** argv) {
    int rank, size;
    int rows, cols;
    double start, end;

    PPM* image;
    PPM* trans;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Arrays for scattering rows
    int* row_cnt = (int*) malloc(sizeof(int) * size);
    int* row_disp = (int*) malloc(sizeof(int) * size);
   
    if (rank == 0) {
        if (argc != 2) {
            errno = EINVAL;
            perror("Error while parsing arguments");
            exit(1);
        }

        printf("Image smoothening using parallel processing\n");
        start = MPI_Wtime();

        // Prepare source image
        const char* fname = argv[1];
        image = read_ppm(fname);
        trans = create_mod_ppm(image, FNAME_MOD);
        rows = image->rows;
        cols = image->cols;
    }
    else {
        image = (PPM*) calloc(1, sizeof(PPM));
        trans = (PPM*) calloc(1, sizeof(PPM));
    }

    // Broadcast rows and cols info
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Define MPI derived data types (pixeltype and rowtype)
    MPI_Datatype pixeltype;
    MPI_Datatype rowtype;
    MPI_Datatype type[] = {MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR};
    int count[] = {1, 1, 1};
    MPI_Aint disp[] = {0, sizeof(unsigned char), sizeof(unsigned char) * 2};

    MPI_Type_create_struct(3, count, disp, type, &pixeltype);
    MPI_Type_commit(&pixeltype);
    MPI_Type_contiguous(cols, pixeltype, &rowtype);
    MPI_Type_commit(&rowtype);

    // Prepare row scattering
    row_cnt[0] = rows / size;
    row_disp[0] = 0;
    for (int i = 1; i < size; i++) {
        row_cnt[i] = rows / size;
        row_disp[i] = (rows / size) * i;
    }
    row_cnt[size - 1] += rows % size;

    pixel* src_buf = (pixel*) calloc(row_cnt[rank] * cols, sizeof(pixel));
    pixel* trans_buf = (pixel*) calloc(row_cnt[rank] * cols, sizeof(pixel));

    // Scatter rows
    MPI_Scatterv(image->data, row_cnt, row_disp, rowtype, src_buf, rows, rowtype, 0, MPI_COMM_WORLD);
    
    // Process each pixel
    for (int i = cols; i < row_cnt[rank] * (cols - 1); i++) {
        smooth(src_buf, trans_buf, row_cnt[rank], cols, i);
    }

    // Gather rows
    MPI_Gatherv(trans_buf, row_cnt[rank], rowtype, trans->data, row_cnt, row_disp, rowtype, 0, MPI_COMM_WORLD);

    // Write processed image to a file and cleanup
    if (rank == 0) {
        // Process the pixels in scatter boundaries
        memcpy(image->data, trans->data, rows * cols * 3);
        for (int k = 0; k < cols; k++) {
            smooth(image->data, trans->data, rows, cols, k);
        }
        for (int i = 1; i < size; i++) {
            for (int j = -1; j <= 0; j++) {
                for (int k = 0; k < cols; k++) {
                    smooth(image->data, trans->data, rows, cols, (row_disp[i] + j) * cols + k);
                }
            }
        }
        for (int k = 0; k < cols; k++) {
            smooth(image->data, trans->data, rows, cols, (rows - 1) * cols + k);
        }

        write_ppm(trans);

        end = MPI_Wtime();
        printf("Elapsed time: %lfms\n", (end - start) * 1000);

        free_ppm(image);
        free_ppm(trans);
    }
    else {
        free(image);
        free(trans);
    }

    MPI_Finalize();

    free(row_cnt);
    free(row_disp);
    free(src_buf);
    free(trans_buf);

    return 0;
}
