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
    ${PROJECT_SOURCE_DIR}/src/elves/othello_montecarlo/
    ${PROJECT_SOURCE_DIR}/src/elves/factorization_montecarlo/
    ${PROJECT_SOURCE_DIR}/src/elves/quadratic_sieve/
    ${MPI_INCLUDE_PATH}
    # Add dependencies' include dirs here
)

OPTION(BUILD_WITH_CUDA "Set to ON to build with cuda elves. (Requires Cuda support.)" ON)

IF(BUILD_WITH_CUDA)
    FIND_PACKAGE(CUDA REQUIRED)
ENDIF()

# Scripts in subdirectories will populate 
# the ELF_LIBRARIES variable
SET(ELF_LIBRARIES "" CACHE STRING INTERNAL FORCE)
ADD_SUBDIRECTORY(elves) 

SET(SUPPORT_LIBRARIES
    ${Boost_LIBRARIES}
    ${MPI_LIBRARIES}
    ${GMP_LIBRARIES}
    ${GMPXX_LIBRARIES}
)

IF(BUILD_WITH_CUDA)
    ADD_DEFINITIONS("-DHAVE_CUDA")

    SET(SUPPORT_LIBRARIES ${SUPPORT_LIBRARIES}
        ${CUDA_UTILS_LIB}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
    )
ENDIF()

ADD_SUBDIRECTORY(common)
ADD_SUBDIRECTORY(main)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(matrixtool)

