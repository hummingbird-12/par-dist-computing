LIBS_SEQ := ppm utils
LIBS_MPI := mpi_ppm mpi_utils
LIBS_H := ppm.h utils.h
LIBS_O := ppm.o utils.o
LIBS_MPI_O := mpi_ppm.o mpi_utils.o

SEQ_OUT := sflip sgray ssmooth
PAR_OUT := pflip pgray psmooth

FLIP := sflip pflip
GRAY := sgray pgray
SMOOTH := ssmooth psmooth

default: $(SEQ_OUT) $(PAR_OUT)

seq: $(SEQ_OUT)

par: $(PAR_OUT)

flip: $(FLIP)

gray: $(GRAY)

smooth: $(SMOOTH)

sflip: $(LIBS_H) sflip.c $(LIBS_SEQ)
	gcc -Wall -c sflip.c -o sflip.o
	gcc -Wall $(LIBS_O) sflip.o -o sflip.out

sgray: $(LIBS_H) sgray.c $(LIBS_SEQ)
	gcc -Wall -c sgray.c -o sgray.o
	gcc -Wall $(LIBS_O) sgray.o -o sgray.out

ssmooth: $(LIBS_H) ssmooth.c $(LIBS_SEQ)
	gcc -Wall -c ssmooth.c -o ssmooth.o
	gcc -Wall $(LIBS_O) ssmooth.o -o ssmooth.out

pflip: $(LIBS_H) pflip.c $(LIBS_MPI)
	mpicc -Wall -c pflip.c -o pflip.o
	mpicc -Wall $(LIBS_MPI_O) pflip.o -o pflip.out

pgray: $(LIBS_H) pgray.c $(LIBS_MPI)
	mpicc -Wall -c pgray.c -o pgray.o
	mpicc -Wall $(LIBS_MPI_O) pgray.o -o pgray.out

psmooth: $(LIBS_H) psmooth.c $(LIBS_MPI)
	mpicc -Wall -c psmooth.c -o psmooth.o
	mpicc -Wall $(LIBS_MPI_O) psmooth.o -o psmooth.out

ppm: ppm.h ppm.c
	gcc -Wall -c ppm.c -o ppm.o

utils: $(LIBS_H) utils.c
	gcc -Wall -c utils.c -o utils.o

mpi_ppm: ppm.h ppm.c
	mpicc -Wall -c ppm.c -o mpi_ppm.o

mpi_utils: $(LIBS_H) utils.c
	mpicc -Wall -c utils.c -o mpi_utils.o

clean:
	rm -rf *.o
	rm -rf *.out

