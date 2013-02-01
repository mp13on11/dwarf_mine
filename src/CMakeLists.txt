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

IF(BUILD_WITH_CUDA)
    FIND_PACKAGE(CUDA REQUIRED)
ENDIF()

# This will generate the variables used
# to set ELV_LIBRARIES below
ADD_SUBDIRECTORY(elves) 

SET(ELF_LIBRARIES
    ${SMP_MATRIX_ELF_LIB}
    ${SHARED_MATRIX_LIB}
    ${SMP_FACTORIZATION_MONTECARLO_ELF_LIB}
    #${SMP_QUADRATIC_SIEVE_ELF_LIB}
)

SET(SUPPORT_LIBRARIES
    ${Boost_LIBRARIES}
    ${MPI_LIBRARIES}
    ${GMP_LIBRARIES}
    ${GMPXX_LIBRARIES}
)

IF(BUILD_WITH_CUDA)
    ADD_DEFINITIONS("-DHAVE_CUDA")
    SET(ELF_LIBRARIES ${ELF_LIBRARIES} 
        ${CUDA_MATRIX_ELF_LIB}
        #${CUDA_QUADRATIC_SIEVE_ELF_LIB}
    )
    SET(SUPPORT_LIBRARIES ${SUPPORT_LIBRARIES}
        ${CUDA_UTILS_LIB}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
    )
ENDIF()

ADD_SUBDIRECTORY(common)
ADD_SUBDIRECTORY(main)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(runner)
ADD_SUBDIRECTORY(matrixtool)

