SET(LIBRARY_NAME cooda-othello-montecarlo)

# Export library for main subproject
REGISTER_ELF_LIB(${LIBRARY_NAME})

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
SET(LINKING_MODE SHARED)
SET(CUDA_PROPAGATE_HOST_FLAGS 0)

CUDA_COMPILE(CUDA_OBJECTS ${CUDA_SOURCES} ${LINKING_MODE} OPTIONS "--generate-code code=sm_21,arch=compute_20 --generate-code code=sm_20,arch=compute_20")

ADD_LIBRARY(${LIBRARY_NAME} ${LINKING_MODE}
    ${GCC_SOURCES} 
    ${CUDA_OBJECTS} 
)
    
ADD_COMMON_MSVC_PROPS(${LIBRARY_NAME})
