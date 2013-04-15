SET(EXECUTABLE_NAME matrixtool)

# Dependencies
INCLUDE_DIRECTORIES(
    # Add dependencies' include dirs here
)

FILE(GLOB AUTO_SOURCES
    "*.h"
    "*.cpp"
)

SET(ADDITIONAL_SOURCES
    ../elves/matrix/MatrixHelper.cpp
    ../elves/matrix/MismatchedMatricesException.cpp
)

# Build executable out of sources
ADD_EXECUTABLE(${EXECUTABLE_NAME} ${AUTO_SOURCES} ${ADDITIONAL_SOURCES})

TARGET_LINK_LIBRARIES(${EXECUTABLE_NAME}
    # Add dependencies' libraries here
    ${MPI_LIBRARIES}
)

ADD_COMMON_MSVC_PROPS(${EXECUTABLE_NAME})
