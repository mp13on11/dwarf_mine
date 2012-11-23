#!/bin/bash

FILTER=$'.*(\.cpp|\.h|CMakeLists.txt)$'
TAB_SIZE=4

echo "Expanding tabs..."

for i in $(grep -lR $'\t' . | grep -E $FILTER)
do
    echo $i
    expand -t $TAB_SIZE $i > $i.notab
    mv $i.notab $i
done

echo "Done."
