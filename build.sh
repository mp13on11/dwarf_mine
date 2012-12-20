#!/bin/bash

BUILD_DIR="build"
GENERATOR="Unix Makefiles"

echo "=== Removing old builds ================================================"
rm -rf $BUILD_DIR
echo
echo "=== Creating empty build directory ====================================="
mkdir $BUILD_DIR
echo 
echo "=== Running out of source build ========================================"
cd $BUILD_DIR
cmake -G "$GENERATOR" ..
echo
echo "=== Compiling =========================================================="
make -j
echo
echo "=== Finished ==========================================================="
