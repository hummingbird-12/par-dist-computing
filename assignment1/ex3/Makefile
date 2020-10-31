LIBS_H := ppm.h utils.h
LIBS_O := ppm.o utils.o
SEQ_OUT := sflip.out sgray.out ssmooth.out

default: $(SEQ_OUT)

sflip.out: $(LIBS_O) sflip.o
	gcc -Wall $(LIBS_O) sflip.o -o sflip.out

sflip.o: $(LIBS_H) sflip.c
	gcc -Wall -c sflip.c -o sflip.o

sgray.out: $(LIBS_O) sgray.o
	gcc -Wall $(LIBS_O) sgray.o -o sgray.out

sgray.o: $(LIBS_H) sgray.c
	gcc -Wall -c sgray.c -o sgray.o

ssmooth.out: $(LIBS_O) ssmooth.o
	gcc -Wall $(LIBS_O) ssmooth.o -o ssmooth.out

ssmooth.o: $(LIBS_H) ssmooth.c
	gcc -Wall -c ssmooth.c -o ssmooth.o

ppm.o: ppm.h ppm.c
	gcc -Wall -c ppm.c -o ppm.o

utils.o: $(LIBS_H) utils.c
	gcc -Wall -c utils.c -o utils.o

clean:
	rm -rf *.o
	rm -rf *.out

