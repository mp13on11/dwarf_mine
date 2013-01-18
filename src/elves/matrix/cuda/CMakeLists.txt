SET(LIBRARY_NAME cuda-matrix)

# Export library for main subproject
SET(CUDA_MATRIX_ELF_LIB ${LIBRARY_NAME} CACHE STRING INTERNAL FORCE)

LINK_DIRECTORIES(
    ${CMAKE_CURRENT_BINARY_DIR}
)
    
INCLUDE_DIRECTORIES(
    # Add dependencies' include dirs here
    ${CUDA_TOOLKIT_INCLUDE}
)

FILE(GLOB GCC_SOURCES
    "*.h"
    "*.cpp"
)

FILE(GLOB CUDA_SOURCES
    "*.h"
    "*.cu"
)

# Build executable out of sources
ENABLE_CPP_11()
SET(LINKING_MODE SHARED)
SET(CUDA_PROPAGATE_HOST_FLAGS 0)

CUDA_COMPILE(CUDA_OBJECTS ${CUDA_SOURCES} ${LINKING_MODE} OPTIONS "--generate-code code=sm_21,arch=compute_20 --generate-code code=sm_20,arch=compute_20 --ptxas-options=-v")

ADD_LIBRARY(${LIBRARY_NAME} ${LINKING_MODE}
    ${GCC_SOURCES} 
    ${CUDA_OBJECTS} 
)

TARGET_LINK_LIBRARIES(${LIBRARY_NAME}
    # Add dependencies' libraries here
    ${SHARED_MATRIX_LIB}
    ${CUDA_UTILS_LIB}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
)    
    
ADD_COMMON_MSVC_PROPS(${LIBRARY_NAME})
