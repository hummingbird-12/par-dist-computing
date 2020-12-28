objects = max_reduction.o divergent.o opt_1.o opt_2.o
app = max_reduction

all: $(objects)
	nvcc -arch=sm_35 $(objects) -o $(app)

%.o: %.cu
	nvcc -x cu -arch=sm_35 -I. -dc -lm $< -o $@

clean:
	rm -f *.o $(app)
