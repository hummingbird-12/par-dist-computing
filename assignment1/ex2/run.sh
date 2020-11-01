#!/bin/bash

if [ "$#" -ne 1 ]; then
    printf "Usage: ./run.sh PROC_NUM\n"
    exit 1
fi

NP=$1
SCAN=scan
BLOCK=block
NBLOCK=nblock
HOSTS=hosts

if [[ ! -f "$SCAN" || ! -f "$BLOCK" || ! -f "$NBLOCK" ]]; then
    printf "Please run the build first: make\n"
    exit 1
fi

if [[ ! -f "$HOSTS" ]]; then
    printf "Hosts file not found\n"
    exit 1
fi

printf "Prefix sums with $NP process(es)\n"

mpiexec -np "$NP" -mca btl ^openib -hostfile "$HOSTS" ./"$SCAN"
echo
sleep 3

mpiexec -np "$NP" -mca btl ^openib -hostfile "$HOSTS" ./"$BLOCK"
echo
sleep 3

mpiexec -np "$NP" -mca btl ^openib -hostfile "$HOSTS" ./"$NBLOCK"

