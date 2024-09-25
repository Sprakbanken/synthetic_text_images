#! /bin/bash

for i in 0 1 2 3 4 5
do
    pdm run python create_dataset.py --partition $i --num_partitions 6 & echo $!
done