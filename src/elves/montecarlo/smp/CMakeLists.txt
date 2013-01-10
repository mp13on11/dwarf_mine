SET(LIBRARY_NAME smp-montecarlo)

# Export library for main subproject
SET(SMP_MONTECARLO_ELF_LIB ${LIBRARY_NAME} CACHE STRING INTERNAL FORCE)

LINK_DIRECTORIES(
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Dependencies
    
INCLUDE_DIRECTORIES(
    # Add dependencies' include dirs here
)

FILE(GLOB GCC_SOURCES
    "*.h"
    "*.cpp"
)

# Build executable out of sources
ENABLE_CPP_11()
SET(LINKING_MODE SHARED)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -funroll-loops")        ## Optimize
endif()

ADD_LIBRARY(${LIBRARY_NAME} ${LINKING_MODE}
    ${GCC_SOURCES} 
)

TARGET_LINK_LIBRARIES(${LIBRARY_NAME}
    ${SHARED_MATRIX_LIB}
)    
    
ADD_COMMON_MSVC_PROPS(${LIBRARY_NAME})
