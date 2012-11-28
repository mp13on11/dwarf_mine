SET(EXECUTABLE_NAME smp)
#SET(BLAS_VENDOR_ATLAS)

FIND_PACKAGE(ATLAS REQUIRED)
    
INCLUDE_DIRECTORIES(
    # Add dependencies' include dirs here
    ${ATLAS_INCLUDES}
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
    ${ATLAS_LIBRARIES}
)    
    
ADD_COMMON_MSVC_PROPS(${EXECUTABLE_NAME})
