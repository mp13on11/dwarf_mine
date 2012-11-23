SET(EXECUTABLE_NAME gold)

# Dependencies
    
INCLUDE_DIRECTORIES(
    # Add dependencies' include dirs here
)

FILE(GLOB AUTO_SOURCES
    "*.h"
    "*.cpp"
)

# Build executable out of sources
ADD_EXECUTABLE(${EXECUTABLE_NAME} ${AUTO_SOURCES})

MESSAGE(STATUS ${BENCHMARK_LIB})

TARGET_LINK_LIBRARIES(${EXECUTABLE_NAME}
    # Add dependencies' libraries here
    ${BENCHMARK_LIB}
    ${TOOLS_LIB}
)

ADD_COMMON_MSVC_PROPS(${EXECUTABLE_NAME})
