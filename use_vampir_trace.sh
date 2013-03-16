#!/bin/sh

if [ "$1" = "-h" ] || [ "$1" != "true" ] && [ "$1" != "false" ] || [ "$2" = "" ]; then
    echo "Usage: {true|false} path/to/CMakeCache.txt.\nUse -h to show this message."
    exit
fi

if [ "$1" = "true" ]; then
    sed -i "s/CMAKE_CXX_COMPILER:FILEPATH=\/usr\/bin\/c++/CMAKE_CXX_COMPILER:FILEPATH=\/usr\/local\/bin\/vtc++/g" "$2"
    sed -i "s/CUDA_NVCC_EXECUTABLE:FILEPATH=\/usr\/local\/cude\/bin\/nvcc/CUDA_NVCC_EXECUTABLE:FILEPATH=\/usr\/local\/bin\/vtnvcc/g" "$2"
    sed -i "s/CMAKE_CXX_FLAGS:STRING=*/CMAKE_CXX_FLAGS:STRING=-vt:inst compinst -vt:hyb /g" "$2"
    sed -i "s/CUDA_NVCC_FLAGS:STRING=*/CUDA_NVCC_FLAGS:STRING=-vt:inst compinst -vt:hyb /g" "$2"
    sed -i "s/CMAKE_CONFIGURATION_TYPES:STRING=*/CMAKE_CONFIGURATION_TYPES:STRING=RelWithDebInfo;/g" "$2"
    sed -i "s/CMAKE_BUILD_TYPE:STRING=*/CMAKE_BUILD_TYPE:STRING=RelWithDebInfo/g" "$2"
elif [ "$1" = "false" ]; then
    sed -i "s/CMAKE_CXX_COMPILER:FILEPATH=\/usr\/local\/bin\/vtc++/CMAKE_CXX_COMPILER:FILEPATH=\/usr\/bin\/c++/g" "$2"
    sed -i "s/CUDA_NVCC_EXECUTABLE:FILEPATH=\/usr\/local\/bin\/vtnvcc/CUDA_NVCC_EXECUTABLE:FILEPATH=\/usr\/local\/cude\/bin\/nvcc/g" "$2"
    sed -i "s/CMAKE_CXX_FLAGS:STRING=-vt:inst compinst -vt:hyb */CMAKE_CXX_FLAGS:STRING=/g" "$2"
    sed -i "s/CUDA_NVCC_FLAGS:STRING=-vt:inst compinst -vt:hyb */CUDA_NVCC_FLAGS:STRING=/g" "$2"
    sed -i "s/CMAKE_CONFIGURATION_TYPES:STRING=RelWithDebInfo;*/CMAKE_CONFIGURATION_TYPES:STRING=/g" "$2"
    sed -i "s/CMAKE_BUILD_TYPE:STRING=RelWithDebInfo/CMAKE_BUILD_TYPE:STRING=/g" "$2"
fi
