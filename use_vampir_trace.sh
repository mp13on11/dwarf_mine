#!/bin/bash
# Kudos to https://rsalveti.wordpress.com/2007/04/03/bash-parsing-arguments-with-getopts/

usage()
{
echo "Usage: $0 yes|no path/to/build/CMakeCache.txt

Sets compilers to VampirTrace compiler wrappers and configures them with flags, or vice-versa."
}

CACHE_FILE=$2
CACHE_DIR=${CACHE_FILE%*CMakeCache.txt}

replace()
{
    sed -i "s/$1/$2/g" "$CACHE_FILE"
}

updateCmake()
{
    cmake $CACHE_DIR
}

if [ "$1" = "yes" ]; then
    replace "CMAKE_CXX_COMPILER:FILEPATH=\/usr\/bin\/c++" "CMAKE_CXX_COMPILER:FILEPATH=\/usr\/local\/bin\/vtc++"
    updateCmake
    replace "CUDA_NVCC_EXECUTABLE:FILEPATH=\/usr\/local\/cuda\/bin\/nvcc" "CUDA_NVCC_EXECUTABLE:FILEPATH=\/usr\/local\/bin\/vtnvcc"
    updateCmake
    replace "CMAKE_CXX_FLAGS:STRING=*" "CMAKE_CXX_FLAGS:STRING=-vt:inst compinst -vt:hyb "
    replace "CUDA_NVCC_FLAGS:STRING=*" "CUDA_NVCC_FLAGS:STRING=-vt:inst compinst -vt:hyb "
    replace "CMAKE_BUILD_TYPE:STRING=*" "CMAKE_BUILD_TYPE:STRING=RelWithDebInfo"
    replace "WARNINGS_AS_ERRORS:BOOL=ON" "WARNINGS_AS_ERRORS:BOOL=OFF"
    updateCmake
    echo "Done integrating VampirTrace configuration."                        
    exit 1
elif [ "$1" = "no" ]; then
    replace "CMAKE_CXX_COMPILER:FILEPATH=\/usr\/local\/bin\/vtc++" "CMAKE_CXX_COMPILER:FILEPATH=\/usr\/bin\/c++"
    updateCmake
    replace "CUDA_NVCC_EXECUTABLE:FILEPATH=\/usr\/local\/bin\/vtnvcc" "CUDA_NVCC_EXECUTABLE:FILEPATH=\/usr\/local\/cuda\/bin\/nvcc"
    updateCmake
    replace "CMAKE_CXX_FLAGS:STRING=-vt:inst compinst -vt:hyb *" "CMAKE_CXX_FLAGS:STRING="
    replace "CUDA_NVCC_FLAGS:STRING=-vt:inst compinst -vt:hyb *" "CUDA_NVCC_FLAGS:STRING="
    replace "CMAKE_BUILD_TYPE:STRING=RelWithDebInfo" "CMAKE_BUILD_TYPE:STRING="
    replace "WARNINGS_AS_ERRORS:BOOL=OFF" "WARNINGS_AS_ERRORS:BOOL=ON"
    updateCmake
    echo "Done removing VampirTrace configuration."
    exit 1
fi

usage
exit
