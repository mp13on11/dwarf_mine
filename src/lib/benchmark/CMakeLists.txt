SET(BENCHMARK_LIB benchmark)

# Export lib name
SET(BENCHMARK_LIB ${BENCHMARK_LIB} PARENT_SCOPE)

# Dependencies
    
INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}
    ${PAPI_INCLUDE_DIR}
    # Add dependencies' include dirs here
)

LINK_DIRECTORIES(
    ${CMAKE_CURRENT_BINARY_DIR}
)

FILE(GLOB AUTO_SOURCES
    "*.h"
    "*.cpp"
)

# Build executable out of sources
ADD_LIBRARY(${BENCHMARK_LIB} SHARED ${AUTO_SOURCES})

MESSAGE(STATUS "Fuuu1" ${PAPI_LIBRARIES})

TARGET_LINK_LIBRARIES(${BENCHMARK_LIB}
    # Add dependencies' libraries here
    ${PAPI_LIBRARIES}
)

ADD_COMMON_MSVC_PROPS(${BENCHMARK_LIB})
