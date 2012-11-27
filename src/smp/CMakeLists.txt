SET(EXECUTABLE_NAME smp)
SET(BLAS_VENDOR_ATLAS)

FIND_PACKAGE(BLAS REQUIRED)
    
INCLUDE_DIRECTORIES(
    # Add dependencies' include dirs here
    ${BLAS_INCLUDE_PATH}
)

FILE(GLOB AUTO_SOURCES
    "*.h"
    "*.cpp"
)

# Build executable out of sources
ADD_EXECUTABLE(${EXECUTABLE_NAME} ${AUTO_SOURCES})

TARGET_LINK_LIBRARIES(${EXECUTABLE_NAME}
    # Add dependencies' libraries here
    ${BENCHMARK_LIB}
    ${TOOLS_LIB}
    ${BLAS_LIBRARIES}
)    
    
ADD_COMMON_MSVC_PROPS(${EXECUTABLE_NAME})
