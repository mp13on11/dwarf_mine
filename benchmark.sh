#!/bin/bash

INA="big_left.txt"
INB="big_right.txt"
OUT="out.txt"

for i in "mpi/mpi-matrix" "cuda/cuda"
do
    echo $i "{"
    echo -ne "\t"
    ("build/src/"$i "--iterations" "1000" $INA $INB $OUT)
    echo "}"
done
