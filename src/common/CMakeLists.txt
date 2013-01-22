SET(LIBRARY_NAME common)

# Export library for main subproject
SET(COMMON_LIBRARY ${LIBRARY_NAME} CACHE STRING INTERNAL FORCE)

# Dependencies
INCLUDE_DIRECTORIES(
    # Add dependencies' include dirs here
)

FILE(GLOB AUTO_SOURCES
    "*.h"
    "*.cpp"
)

# Build executable out of sources
SET(LINKING_MODE SHARED)
ADD_LIBRARY(${LIBRARY_NAME} ${LINKING_MODE}
    ${AUTO_SOURCES}
)

ADD_COMMON_MSVC_PROPS(${LIBRARY_NAME})
