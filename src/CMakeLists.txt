# Add global dependencies here
FIND_PACKAGE(Boost REQUIRED)
FIND_PACKAGE(PAPI REQUIRED)

INCLUDE(Utils)

INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/
    ${PROJECT_SOURCE_DIR}/src/lib/
    # Add dependencies' include dirs here
)

OPTION(BUILD_CUDA_EXECUTABLES "Set to ON to build the CUDA executables (requires CUDA -- obviously)." ON)
OPTION(BUILD_MPI_EXECUTABLES "Set to ON to build the MPI executables (requires MPI -- should be included in boost...)." ON)
OPTION(BUILD_SMP_EXECUTABLES "Set to ON to build the SMP executables (requires ATLAS)." ON)

ADD_SUBDIRECTORY(lib/benchmark)
ADD_SUBDIRECTORY(lib/tools)
ADD_SUBDIRECTORY(gold)
ADD_SUBDIRECTORY(test)
IF(BUILD_CUDA_EXECUTABLES)
    ADD_SUBDIRECTORY(cuda)
ENDIF()
IF(BUILD_MPI_EXECUTABLES)
    ADD_SUBDIRECTORY(mpi)
ENDIF()
IF(BUILD_SMP_EXECUTABLES)
    ADD_SUBDIRECTORY(smp)
ENDIF()
