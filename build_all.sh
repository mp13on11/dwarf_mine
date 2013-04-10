#!/bin/bash

BUILD_DIR="build"
BUILD_TYPE="Debug"
GENERATOR="Unix Makefiles"
SOURCE_DIR=`pwd`

while getopts "d:t:" OPTION
do
	case $OPTION in
		d)
			BUILD_DIR="$OPTARG"
			;;
		t)	
			BUILD_TYPE="$OPTARG"
			;;
		?)
			cat << EOF
usage: $0 options
OPTIONAL OPTIONS:
  -d Specify the build directory (default build)
  -t Specify the build type accepted by cmake (default debug)
EOF
			exit
			;;
	esac
done				

echo "=== Removing old builds == rm -rf $BUILD_DIR ==="
rm -rf $BUILD_DIR
echo "=== Creating empty build directory == mkdir $BUILD_DIR ==="
mkdir -p $BUILD_DIR
echo "=== Running out of source build == Build: $BUILD_TYPE in $BUILD_DIR ==="
cd $BUILD_DIR
cmake -G "$GENERATOR" "$SOURCE_DIR"
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" .
echo "=== Compiling == make -j ==="
make -j
echo "=== Finished ==="
