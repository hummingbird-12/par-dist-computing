# Sample MPI program

## Example `hosts` file
```
cspro2.sogang.ac.kr slots=3
cspro4.sogang.ac.kr slots=3
cspro5.sogang.ac.kr slots=3
```

## Compile
```bash
mpicc hello.c -o hello
```

## Execute

Run program `hello` using 9 processes using hosts specified in `hosts` file.
```bash
mpiexec -np 9 -mca btl ^openib -hostfile hosts ./hello
```

