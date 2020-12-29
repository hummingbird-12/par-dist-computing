Problem 2 - CUDA Programming 1
------------------------------

* Instructions
1. Build using the `Makefile`
   $ make

2. Execute the program
   $ ./matmul BLOCK_SIZE

   Example:
   $ ./matmul 256

* Notes
- The development was done within Google Colaboratory's GPU environment.
  Use this template to open Google Colaboratory in Visual Studio Code server:
  https://colab.research.google.com/github/hummingbird-12/par-dist-computing/blob/main/assignment3/Colabcode%20template.ipynb
- The program has been developed with `nvcc` using `-arch=sm_35` option.
- In the program output:
   `matmul_global` -> implementation using global memory
   `matmul_shared` -> implementation using shared memory
   `matmul_optim`  -> implementation further optimized

* Developer
ID: 20161577
Name: Inho Kim
