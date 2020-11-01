#!/bin/bash

if [ "$#" -ne 3 ]; then
    printf "Usage: ./run.sh flip|gray|smooth PROC_NUM PPM_FILE\n"
    exit 1
fi

NP=$2
HOSTS=hosts

if [[ ! "$1" = "flip" && ! "$1" = "gray" && ! "$1" = "smooth" ]]; then
    printf "Invalid mode\n"
    exit 1
fi

if [[ ! -f "s${1}.out" || ! -f "p${1}.out" ]]; then
    printf "Please run the build first: make $1\n"
    exit 1
fi

if [[ ! -f "$3" ]]; then
    printf "PPM file not found\n"
    exit 1
fi

if [[ ! -f "$HOSTS" ]]; then
    printf "Hosts file not found\n"
    exit 1
fi

./s${1}.out "$3"
echo
sleep 3

mpiexec -np "$NP" -mca btl ^openib -hostfile "$HOSTS" ./p${1}.out "$3"

