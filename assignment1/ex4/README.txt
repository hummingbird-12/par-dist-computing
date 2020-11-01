Exercise 4 - RPC Calculator
---------------------------


* Content
___
 |-- calc.x         # RPC contract
 |-- calc_server.c  # Server-side program
 |-- calc_svc.c     # Server-side stub
 |-- calc_client.c  # Client-side program
 |-- calc_clnt.c    # Client-side stub
 |-- calc_xdr.c     # eXternal Data Representation
 |-- calc.h         # Header
 |-- Makefile.calc  # Generated Makefile
 |-- Makefile       # Build commands
 |-- README.txt

* Instructions
1. Build using the `Makefile`
   $ make

2. Execute the server program
   $ ./calc_server

3. Execute the client program
   $ ./calc_client HOST_ADDR

   Example:
   $ ./calc_client cspro2.sogang.ac.kr

* Developer
ID: 20161577
Name: Inho Kim

