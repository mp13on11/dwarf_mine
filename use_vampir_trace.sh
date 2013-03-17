#!/bin/bash
# Kudos to https://rsalveti.wordpress.com/2007/04/03/bash-parsing-arguments-with-getopts/

usage()
{
echo "Usage: $0 options

Sets compilers to VampirTrace compiler wrappers and configures them with flags, or vice-versa.

OPTIONS:
    -h  Show this message
    -s  Configure with or without VampirTrace (yes|no)
    -f  Path to build directory's CMakeCache.txt"
}

INTEGRATE=
CACHE_FILE=
CACHE_DIR=

while getopts "h:s:f:*" OPTION
do
    case $OPTION in
        h)  
            usage
            exit 1
            ;;
        s)  
            INTEGRATE=$OPTARG
            ;;
        f)  
            CACHE_FILE=$OPTARG
            CACHE_DIR=${CACHE_FILE%*CMakeCache.txt}
            ;;
        *)  
            usage
            exit
            ;;
    esac
done
\/usr\/local\/cuda\/bin\/nvcc
if [ "$INTEGRATE" = "yes" ]; then
    sed -i "s/CMAKE_CXX_COMPILER:FILEPATH=\/usr\/bin\/c++/CMAKE_CXX_COMPILER:FILEPATH=\/usr\/local\/bin\/vtc++/g" "$CACHE_FILE"
    cmake $CACHE_DIR
    sed -i "s/CUDA_NVCC_EXECUTABLE:FILEPATH=\/usr\/local\/cuda\/bin\/nvcc/CUDA_NVCC_EXECUTABLE:FILEPATH=\/usr\/local\/bin\/vtnvcc/g" "$CACHE_FILE"
    cmake $CACHE_DIR
    sed -i "s/CMAKE_CXX_FLAGS:STRING=*/CMAKE_CXX_FLAGS:STRING=-vt:inst compinst -vt:hyb /g" "$CACHE_FILE"
    sed -i "s/CUDA_NVCC_FLAGS:STRING=*/CUDA_NVCC_FLAGS:STRING=-vt:inst compinst -vt:hyb /g" "$CACHE_FILE"
    sed -i "s/CMAKE_BUILD_TYPE:STRING=*/CMAKE_BUILD_TYPE:STRING=RelWithDebInfo/g" "$CACHE_FILE"
    sed -i "s/WARNINGS_AS_ERRORS:BOOL=ON/WARNINGS_AS_ERRORS:BOOL=OFF/g" "$CACHE_FILE"
    cmake $CACHE_DIR
    echo "Done integrating VampirTrace configuration."
    exit 1
elif [ "$INTEGRATE" = "no" ]; then
    sed -i "s/CMAKE_CXX_COMPILER:FILEPATH=\/usr\/local\/bin\/vtc++/CMAKE_CXX_COMPILER:FILEPATH=\/usr\/bin\/c++/g" "$CACHE_FILE"
    cmake $CACHE_DIR
    sed -i "s/CUDA_NVCC_EXECUTABLE:FILEPATH=\/usr\/local\/bin\/vtnvcc/CUDA_NVCC_EXECUTABLE:FILEPATH=\/usr\/local\/cuda\/bin\/nvcc/g" "$CACHE_FILE"
    cmake $CACHE_DIR
    sed -i "s/CMAKE_CXX_FLAGS:STRING=-vt:inst compinst -vt:hyb */CMAKE_CXX_FLAGS:STRING=/g" "$CACHE_FILE"
    sed -i "s/CUDA_NVCC_FLAGS:STRING=-vt:inst compinst -vt:hyb */CUDA_NVCC_FLAGS:STRING=/g" "$CACHE_FILE"
    sed -i "s/CMAKE_BUILD_TYPE:STRING=RelWithDebInfo/CMAKE_BUILD_TYPE:STRING=/g" "$CACHE_FILE"
    sed -i "s/WARNINGS_AS_ERRORS:BOOL=OFF/WARNINGS_AS_ERRORS:BOOL=ON/g" "$CACHE_FILE"
    cmake $CACHE_DIR
    echo "Done removing VampirTrace configuration."
    exit 1
fi

usage
exit
