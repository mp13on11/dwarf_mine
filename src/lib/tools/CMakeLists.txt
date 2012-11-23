SET(LIB_NAME libtools)

# Dependencies
    
INCLUDE_DIRECTORIES(
    # Add dependencies' include dirs here
)

FILE(GLOB AUTO_SOURCES
    "*.h"
    "*.cpp"
)

# Build executable out of sources
ADD_LIBRARY(${LIB_NAME} SHARED ${AUTO_SOURCES})

TARGET_LINK_LIBRARIES(${LIB_NAME}
    # Add dependencies' libraries here
)    
    
ADD_COMMON_MSVC_PROPS(${LIB_NAME})
