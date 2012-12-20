SET(EXECUTABLE_NAME test)

ENABLE_CPP_11()

# Dependencies
    
INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}/gtest-1.6.0/include/
    # Add dependencies' include dirs here
)

FILE(GLOB AUTO_SOURCES
    "*.h"
    "*.cpp"
)

# Build executable out of sources
ADD_EXECUTABLE(${EXECUTABLE_NAME} ${AUTO_SOURCES})

TARGET_LINK_LIBRARIES(${EXECUTABLE_NAME}
    # Add dependencies' libraries here
    ${TOOLS_LIB}
    "gtest"
)    
    
ADD_COMMON_MSVC_PROPS(${EXECUTABLE_NAME})
