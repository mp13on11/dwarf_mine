SET(TOOLS_LIB tools)

# Dependencies
    
INCLUDE_DIRECTORIES(
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
ADD_LIBRARY(${TOOLS_LIB} SHARED ${AUTO_SOURCES})

TARGET_LINK_LIBRARIES(${TOOLS_LIB}
    # Add dependencies' libraries here
)    
    
ADD_COMMON_MSVC_PROPS(${TOOLS_LIB})
