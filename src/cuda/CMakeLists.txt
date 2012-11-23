SET(EXECUTABLE_NAME cuda)

# Dependencies
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(Boost REQUIRED)
    
INCLUDE_DIRECTORIES(
    # Add dependencies' include dirs here
)

FILE(GLOB AUTO_SOURCES
    "*.h"
    "*.cpp"
)

# Build executable out of sources
CUDA_ADD_EXECUTABLE(${EXECUTABLE_NAME} ${AUTO_SOURCES})
CUDA_ADD_CUBLAS_TO_TARGET(${EXECUTABLE_NAME})

TARGET_LINK_LIBRARIES(${EXECUTABLE_NAME}
    # Add dependencies' libraries here
    ${BENCHMARK_LIB}
    ${TOOLS_LIB}
)    
    
ADD_COMMON_MSVC_PROPS(${EXECUTABLE_NAME})
