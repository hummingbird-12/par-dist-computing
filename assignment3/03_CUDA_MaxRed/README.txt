Problem 3 - CUDA Programming 2
------------------------------

* Instructions
1. Build using the `Makefile`
   $ make

2. Execute the program
   $ ./max_reduction BLOCK_SIZE [ARRAY_SIZE]

   Example:
   $ ./max_reduction 256
   $ ./max_reduction 128 5000

* Notes
- The development was done within Google Colaboratory's GPU environment.
  Use this template to open Google Colaboratory in Visual Studio Code server:
  https://colab.research.google.com/github/hummingbird-12/par-dist-computing/blob/main/assignment3/Colabcode%20template.ipynb
- The program has been developed with `nvcc` using `-arch=sm_35` option.
- In the program output:
   `reduction_sequential` -> sequential implementation
   `reduction_divergent`  -> parallel implementation without considering path divergence
   `reduction_opt_1`      -> parallel implementation with reduced path divergence
   `reduction_opt_2`      -> parallel implementation with reduced path divergence, shared memory

* Developer
ID: 20161577
Name: Inho Kim
