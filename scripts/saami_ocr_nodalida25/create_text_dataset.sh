#! /bin/bash

shopt -s globstar

root_dir=$(git rev-parse --show-toplevel)
data_dir=${root_dir}/input/saami_ocr_nodalida25
raw_dir=${data_dir}/raw
for i in $raw_dir/corpus-sm*/converted/**/*.xml
do
    file_path=$(realpath --relative-to="$raw_dir" "$i")
    corpus=$(echo "$file_path" | cut -d'/' -f1)
    new_name=$(echo "$file_path" | cut -d'/' -f2- | sed 's/\//_/g').lines.txt

    echo $corpus/$new_name

    new_parent="$data_dir/lines/$corpus"
    mkdir -p "$new_parent"
    pdm run ccat $i -a | sed 's/¶$//' > "$new_parent/$new_name"
done
