SET(LIBRARY_NAME smp-quadratic-sieve)

# Export library for main subproject
REGISTER_ELF_LIB(${LIBRARY_NAME})

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
SET(LINKING_MODE STATIC)

IF(CMAKE_COMPILER_IS_GNUCXX OR USE_VAMPIR_TRACE)
    ADD_CXX_FLAGS("-O3 -fopenmp -funroll-loops")
ENDIF()

ADD_LIBRARY(${LIBRARY_NAME} ${LINKING_MODE}
    ${GCC_SOURCES} 
)
    
ADD_COMMON_MSVC_PROPS(${LIBRARY_NAME})
