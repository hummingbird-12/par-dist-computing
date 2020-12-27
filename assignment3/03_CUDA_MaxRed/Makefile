objects = max_reduction.o divergent.o
app = max_reduction

all: $(objects)
	nvcc -arch=sm_30 $(objects) -o $(app)

%.o: %.cu
	nvcc -x cu -arch=sm_30 -I. -dc -lm $< -o $@

clean:
	rm -f *.o *.txt $(app)

