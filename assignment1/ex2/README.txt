Exercise 2 - Prefix sums
------------------------


* Content
___
 |-- scan.c     # Implementation using `MPI_Scan()`
 |-- block.c    # Implementation using blocking send/recv
 |-- nblock.c   # Implementation using non-blocking send/recv
 |-- hosts      # Hosts list for MPI
 |-- Makefile   # Build commands
 |-- run.sh     # Run script
 |-- README.txt

* Instructions
1. Build using the `Makefile`
   $ make

2. Execute using the `run.sh` script
   Usage:
   $ ./run.sh PROC_NUM
   - PROC_NUM : number of process(es)

   Example:
   $ ./run.sh 16

* Developer
ID: 20161577
Name: Inho Kim

