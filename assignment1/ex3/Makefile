LIBS := ppm utils
LIBS_H := ppm.h utils.h
LIBS_O := ppm.o utils.o
LIBS_MPI_O := mpi_ppm.o mpi_utils.o

SEQ_OUT := sflip sgray ssmooth
PAR_OUT := pflip pgray # psmooth

default: $(SEQ_OUT) $(PAR_OUT)

seq: $(SEQ_OUT)

par: $(PAR_OUT)

sflip: $(LIBS_H) sflip.c $(LIBS)
	gcc -Wall -c sflip.c -o sflip.o
	gcc -Wall $(LIBS_O) sflip.o -o sflip.out

sgray: $(LIBS_H) sgray.c $(LIBS)
	gcc -Wall -c sgray.c -o sgray.o
	gcc -Wall $(LIBS_O) sgray.o -o sgray.out

ssmooth: $(LIBS_H) ssmooth.c $(LIBS)
	gcc -Wall -c ssmooth.c -o ssmooth.o
	gcc -Wall $(LIBS_O) ssmooth.o -o ssmooth.out

pflip: $(LIBS_H) pflip.c $(LIBS)
	mpicc -Wall -c pflip.c -o pflip.o
	mpicc -Wall $(LIBS_MPI_O) pflip.o -o pflip.out

pgray: $(LIBS_H) pgray.c $(LIBS)
	mpicc -Wall -c pgray.c -o pgray.o
	mpicc -Wall $(LIBS_MPI_O) pgray.o -o pgray.out

ppm: ppm.h ppm.c
	gcc -Wall -c ppm.c -o ppm.o
	mpicc -Wall -c ppm.c -o mpi_ppm.o

utils: $(LIBS_H) utils.c
	gcc -Wall -c utils.c -o utils.o
	mpicc -Wall -c utils.c -o mpi_utils.o

clean:
	rm -rf *.o
	rm -rf *.out

