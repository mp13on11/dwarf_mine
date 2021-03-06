SET(LIBRARY_NAME smp-othello-montecarlo)

# Export library for main subproject
REGISTER_ELF_LIB(${LIBRARY_NAME})

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
SET(LINKING_MODE SHARED)

if(CMAKE_COMPILER_IS_GNUCXX OR USE_VAMPIR_TRACE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -funroll-loops")        ## Optimize
endif()

ADD_LIBRARY(${LIBRARY_NAME} ${LINKING_MODE}
    ${GCC_SOURCES} 
)

TARGET_LINK_LIBRARIES(${LIBRARY_NAME}
    ${SHARED_OTHELLO_MONTECARLO_LIB}
)    
    
ADD_COMMON_MSVC_PROPS(${LIBRARY_NAME})
