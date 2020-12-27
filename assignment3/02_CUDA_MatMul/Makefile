objects = matmul.o matmul_global.o matmul_shared.o matmul_optimized.o
app = matmul

all: $(objects)
	nvcc -arch=sm_30 $(objects) -o $(app)

%.o: %.cu
	nvcc -x cu -arch=sm_30 -I. -dc $< -o $@

clean:
	rm -f *.o *.txt $(app)

