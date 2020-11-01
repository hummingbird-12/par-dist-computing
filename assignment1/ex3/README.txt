Exercise 3 - Image transformation
---------------------------------


* Content
___
 |-- ppm_example/   # Sample PPM images
 |-- sflip.c        # Flip - sequential implementation
 |-- pflip.c        # Flip - parallel implementation
 |-- sgray.c        # Grayscale - sequential implementation
 |-- pgray.c        # Grayscale - parallel implementation
 |-- ssmooth.c      # Smoothen - sequential implementation
 |-- psmooth.c      # Smoothen - parallel implementation
 |-- ppm.*          # Library
 |-- utils.*        # Library
 |-- hosts          # Hosts list for MPI
 |-- Makefile       # Build commands
 |-- run.sh         # Run script
 |-- README.txt

* Instructions
1. Build using the `Makefile`
   $ make

   Or instead:
   $ make seq       # Build all sequential implementations
   $ make par       # Build all parallel implementations
   $ make flip      # Build all flip transformations
   $ make gray      # Build all grayscale transformations
   $ make smooth    # Build all smoothen transformations

2. Prepare PPM files
   Copy PPM file(s) to transform into current directory.

   Example:
   $ cp ppm_example/Iggy.1024.ppm .

3. Execute using the `run.sh` script
   Usage:
   $ ./run.sh flip|gray|smooth PROC_NUM PPM_FILE
   - flip|gray|smooth : transformation mode
   - PROC_NUM : number of process(es)
   - PPM_FILE : PPM file to transform

   Example:
   $ ./run.sh flip 16 Iggy.1024.ppm

* Developer
ID: 20161577
Name: Inho Kim

