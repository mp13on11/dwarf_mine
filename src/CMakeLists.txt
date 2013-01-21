# Add global dependencies here

SET(Boost_USE_STATIC_LIBS        ON)
SET(Boost_USE_MULTITHREADED      ON)
SET(Boost_USE_STATIC_RUNTIME    OFF)
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options)

FIND_PACKAGE(MPI REQUIRED)
FIND_PACKAGE(GMP REQUIRED)

# Globally available include directories
INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/
    ${PROJECT_SOURCE_DIR}/src/elves/
    ${PROJECT_SOURCE_DIR}/src/elves/matrix/
    ${MPI_INCLUDE_PATH}
    # Add dependencies' include dirs here
)

OPTION(BUILD_WITH_CUDA "Set to ON to build with cuda elves. (Requires Cuda support.)" ON)

# This will generate the variables used
# to set ELV_LIBRARIES below
ADD_SUBDIRECTORY(elves) 

SET(ELF_LIBRARIES
    ${SMP_MATRIX_ELF_LIB}
    ${SHARED_MATRIX_LIB}
    ${SMP_FACTORIZE_ELF_LIB}
    ${SHARED_FACTORIZE_LIB}
)

IF(BUILD_WITH_CUDA)
    ADD_DEFINITIONS("-DHAVE_CUDA")
    SET(ELF_LIBRARIES ${ELF_LIBRARIES} 
        ${CUDA_MATRIX_ELF_LIB}
        ${CUDA_FACTORIZE_ELF_LIB}
    )
ENDIF()

ADD_SUBDIRECTORY(main)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(matrixtool)

