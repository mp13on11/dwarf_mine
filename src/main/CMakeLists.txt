SET(EXECUTABLE_NAME dwarf_mine)

ENABLE_CPP_11()

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

IF(BUILD_WITH_CUDA)
    TARGET_LINK_LIBRARIES(${EXECUTABLE_NAME}
        # Add dependencies' libraries here
        ${CUDA_MATRIX_ELF_LIB}
        ${SMP_MATRIX_ELF_LIB}
        ${MPI_LIBRARIES}
    )
ELSE()
    TARGET_LINK_LIBRARIES(${EXECUTABLE_NAME}
        # Add dependencies' libraries here
        ${SMP_MATRIX_ELF_LIB}
        ${MPI_LIBRARIES}
    )
ENDIF()

ADD_COMMON_MSVC_PROPS(${EXECUTABLE_NAME})
