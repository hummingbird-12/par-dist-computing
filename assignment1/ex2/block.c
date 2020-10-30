#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mpi.h"

#define MAX_VAL (1 << 10)

int main(int argc, char **argv) {
    int rank, size, data, prev_result, result;
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        srand(time(NULL) + rank);
        printf("Prefix sums of %d random integers using `MPI_Scan()`\n", size);
        start = MPI_Wtime();

        // Generate random values
        int* val = (int*) malloc(sizeof(int) * size);
        for (int i = 0; i < size; i++) {
            // val[i] = rand() % MAX_VAL;
            val[i] = i;
        }

        // Scatter data
        data = val[0];
        for (int i = 1; i < size; i++) {
            MPI_Send(val + i, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        prev_result = 0;
    }
    else {
        MPI_Status stat;

        // Receive scattered data
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);

        // Receive previous prefix sum
        MPI_Recv(&prev_result, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &stat);
    }

    // Calculate prefix sum and send to next process
    result = prev_result + data;
    if (rank < size - 1) {
        MPI_Send(&result, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD);
    }
    printf("[Process %2d]\tValue = %d\tPrefix_sum = %d\n", rank, data, result);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        end = MPI_Wtime();
        printf("Elapsed time: %lfms\n", (end - start) * 1000);
    }

    MPI_Finalize();

    return 0;
}

