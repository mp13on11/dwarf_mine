SET(EXECUTABLE_NAME dwarf_mine)

ENABLE_CPP_11()

FIND_PACKAGE(MPI REQUIRED)

# Dependencies
    
INCLUDE_DIRECTORIES(
    # Add dependencies' include dirs here
    ${MPI_INCLUDE_PATH}
)

FILE(GLOB AUTO_SOURCES
    "*.h"
    "*.cpp"
)

# Build executable out of sources
ADD_EXECUTABLE(${EXECUTABLE_NAME} ${AUTO_SOURCES})

TARGET_LINK_LIBRARIES(${EXECUTABLE_NAME}
    # Add dependencies' libraries here
    ${CUDA_MATRIX_ELF_LIB}
    ${SMP_MATRIX_ELF_LIB}
    ${MPI_LIBRARIES}
)

ADD_COMMON_MSVC_PROPS(${EXECUTABLE_NAME})
