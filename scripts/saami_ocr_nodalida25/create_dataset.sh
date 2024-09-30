#! /bin/bash

if [ -z "$1" ]; then
  echo "Error: No language provided."
  exit 1
fi

mkdir -p /mnt/disk3/synthetic_ocr_data/
num_partitions=25
for i in $(seq 0 $((num_partitions-1)))
do
    pdm run python create_dataset.py --language $1 --output_dir "/mnt/disk3/synthetic_ocr_data/" --partition $i --num_partitions $num_partitions & echo $!
done