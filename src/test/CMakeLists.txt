SET(EXECUTABLE_NAME test)

# Dependencies
#FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(Boost REQUIRED)
    
INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}
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
    "libtools"
    "gtest"
)    
    
IF(MSVC)
    ADD_DEFINITIONS(/D_CRT_SECURE_NO_WARNINGS /DNOMINMAX /DWIN32_LEAN_AND_MEAN)

    SET_TARGET_PROPERTIES(${EXECUTABLE_NAME}
        PROPERTIES
        LINK_FLAGS "/NODEFAULTLIB:libcmt /SUBSYSTEM:CONSOLE")
ENDIF(MSVC)
