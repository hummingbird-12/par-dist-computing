objects = matmul.o matmul_global.o
app = matmul

all: $(objects)
	nvcc -arch=sm_20 $(objects) -o $(app)

%.o: %.cu
	nvcc -x cu -arch=sm_20 -I. -dc $< -o $@

clean:
	rm -f *.o *.txt $(app)

