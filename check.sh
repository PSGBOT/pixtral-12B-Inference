#!/bin/bash

# execute this script in the dataset folder

# check mask index
find . -type d -exec bash -c '[ ! -e "$0/mask0.png" ] && echo "$0"' {} \;

# check too many masks
find . -type d -exec bash -c 'shopt -s nullglob dotglob; count=("$1"/*); [ "${#count[@]}" -gt 128 ] && echo "$1 (${#count[@]})"' _ {} \;

# expected output:
# .
#./val
#./train
#./val (3028)
#./train (12415)
