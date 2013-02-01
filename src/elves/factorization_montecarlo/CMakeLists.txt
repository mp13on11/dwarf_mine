SET(LIBRARY_NAME smp-factorization-montecarlo)

# Export library for main subproject
SET(SMP_FACTORIZATION_MONTECARLO_ELF_LIB ${LIBRARY_NAME} CACHE STRING INTERNAL FORCE)

# Dependencies
LINK_DIRECTORIES(
    ${CMAKE_CURRENT_BINARY_DIR}
)
    
INCLUDE_DIRECTORIES(
    # Add dependencies' include dirs here
)

FILE(GLOB GCC_SOURCES
    "*.h"
    "*.cpp"
)

# Build executable out of sources
SET(LINKING_MODE SHARED)

IF(CMAKE_COMPILER_IS_GNUCXX)
    ADD_CXX_FLAGS("-O3 -fopenmp -funroll-loops")        ## Optimize
ENDIF()

ADD_LIBRARY(${LIBRARY_NAME} ${LINKING_MODE}
    ${GCC_SOURCES} 
)
    
ADD_COMMON_MSVC_PROPS(${LIBRARY_NAME})
