# Add global dependencies here
FIND_PACKAGE(Boost REQUIRED)
FIND_PACKAGE(MPI REQUIRED)

INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/
    ${PROJECT_SOURCE_DIR}/src/elves/
    ${MPI_INCLUDE_PATH}
    # Add dependencies' include dirs here
)

OPTION(BUILD_WITH_CUDA "Set to ON to build with cuda elves. (Requires Cuda support.)" ON)

IF(BUILD_WITH_CUDA)
    ADD_DEFINITIONS("-DHAVE_CUDA")
ENDIF()

ADD_SUBDIRECTORY(elves)
ADD_SUBDIRECTORY(main)
