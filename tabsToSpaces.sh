#!/bin/bash

FILTER=$'.*(\.cpp|\.h|CMakeLists.txt)$'
TAB_SIZE=4

echo "Checking for tabs in files..."
FILES=$(grep -lR $'\t' . | grep -E $FILTER)
if [ $? -eq 0 ];
then
    echo "Found files, expanding tabs..."
else
    echo "No tabs in files. Stopping."
    exit 1
fi

for i in $FILES
do
    echo $i
    expand -t $TAB_SIZE $i > $i.notab
    mv $i.notab $i
done

echo "Done."
